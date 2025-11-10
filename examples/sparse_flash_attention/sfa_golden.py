import torch
import numpy as np
def pa_to_bsnd(pa_in, block_table, actual_seq_lengths):
    block_num, block_size, n, d = pa_in.shape
    b = len(actual_seq_lengths)
    output = torch.zeros((b, block_num * block_size // b, 1, d)).to(pa_in.dtype)
    for i in range(b):
        for j in range(actual_seq_lengths[i] // block_size):
            output[i, j * block_size: (j + 1) * block_size, 0, :] = \
                pa_in[block_table[i][j], :, 0, :].reshape(block_size, d)
    return output

def gather_kv(k_tensor, v_tensor, sparse_indices, sparse_block_size, sparse_count, batch, n2_idx, s1_idx,
             cur_actual_seq_lengths_kv):
    s2_sparse = list()
    for sparse_id in sparse_indices:
        if sparse_id == -1:
            break
        begin_idx = sparse_id * sparse_block_size
        end_idx = begin_idx + sparse_block_size \
                if begin_idx + sparse_block_size <= cur_actual_seq_lengths_kv else cur_actual_seq_lengths_kv
        s2_sparse.extend(np.arange(begin_idx, end_idx))

    k_sparse, v_sparse = k_tensor[batch, n2_idx, s2_sparse, :], v_tensor[batch, n2_idx, s2_sparse, :]

    return k_sparse, v_sparse

def mask(res, cur_actual_seq_q, cur_actual_seq, topk_indices, s1_idx, sparse_blocksize):
    # 求尾块ID和尾块长度
    sparse_tail_idx = np.ceil(cur_actual_seq / sparse_blocksize)
    sparse_tail_seq_len = cur_actual_seq % sparse_blocksize
    if sparse_tail_seq_len == 0:
        sparse_tail_seq_len = sparse_blocksize

    delta_s = cur_actual_seq - cur_actual_seq_q
    threshold = delta_s + s1_idx + 1
    s_idx = 0
    for _, sparse_id in enumerate(topk_indices):
        if sparse_id == -1:
            break
        begin_idx = sparse_id * sparse_blocksize
        block_len = sparse_blocksize if sparse_id != sparse_tail_idx - 1 else sparse_tail_seq_len
        end_idx = begin_idx + block_len
        if begin_idx < threshold and end_idx <= threshold:
            s_idx += block_len
            continue
        elif end_idx > threshold:
            local_offset = 0 if threshold <= begin_idx else threshold - begin_idx
            mask_begin = s_idx + local_offset
            mask_end = s_idx + block_len
            res[:, mask_begin: mask_end] = -1e12
        s_idx += block_len

    return res

def softmax(x):
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y / x_sum
    return ans

def cpu_sparse_flash_attention(
    query, key, value, sparse_indices, scale_value, sparse_block_size,
    actual_seq_lengths_query, actual_seq_lengths_kv,
    query_rope=None, key_rope=None,
    layout_query='BSND', layout_kv='BSND', sparse_mode=3, block_table=None):
    query = query.cpu().to(torch.float32).numpy()
    if layout_kv == 'PA_BSND':
        key = pa_to_bsnd(key, block_table, actual_seq_lengths_kv)
        key_rope = pa_to_bsnd(key_rope, block_table, actual_seq_lengths_kv)
    key = key.cpu().to(torch.float32).numpy()
    value = key.copy()
    sparse_indices = sparse_indices.cpu().numpy()
    actual_seq_lengths_query = actual_seq_lengths_query.cpu().numpy()
    actual_seq_lengths_kv = actual_seq_lengths_kv.cpu().numpy()
    query_rope = query_rope.cpu().to(torch.float32).numpy()
    key_rope = key_rope.cpu().to(torch.float32).numpy()
    batch_size = actual_seq_lengths_query.shape[0]
    num_heads = query.shape[2]
    num_kv_heads = key.shape[2]
    sparse_count = sparse_indices.shape[-1]
    g = num_heads // num_kv_heads

    if layout_query == 'TND':
        actual_seq_lengths_query = trans_tnd_actseq(actual_seq_lengths_query)
        query = trans_tnd_to_bsnd(query, query.shape, actual_seq_lengths_query)
        query_rope = trans_tnd_to_bsnd(query_rope, query_rope.shape, actual_seq_lengths_query)
        sparse_indices = trans_tnd_to_bsnd(sparse_indices, sparse_indices.shape, actual_seq_lengths_query)

    q_bnsd_tensor = np.transpose(np.concatenate([query, query_rope], axis=-1), axes=(0, 2, 1, 3))
    k_bnsd_tensor = np.transpose(np.concatenate([key, key_rope], axis=-1), axes=(0, 2, 1, 3))
    v_bnsd_tensor = np.transpose(value, axes=(0, 2, 1, 3))
    sparse_indices = np.transpose(sparse_indices, axes=(0, 2, 1, 3))
    matmul_dtype = np.float32
    out_shape_bnsd = list(q_bnsd_tensor.shape)
    out_shape_bnsd[-1] = out_shape_bnsd[-1] - query_rope.shape[-1]
    y = np.zeros(out_shape_bnsd, dtype=np.float32)

    for batch in range(batch_size):
        cur_acutal_seq_lengths_q = actual_seq_lengths_query[batch]
        cur_actual_seq_lengths_kv = actual_seq_lengths_kv[batch]
        for n2_idx in range(num_kv_heads):
            for s1_idx in range(cur_acutal_seq_lengths_q):
                q_curr = q_bnsd_tensor[batch, n2_idx * g: (n2_idx + 1) * g, s1_idx, :]
                cur_sparse_indices = sparse_indices[batch, n2_idx, s1_idx, :]
                k_sparse, v_sparse = gather_kv(k_bnsd_tensor, v_bnsd_tensor, cur_sparse_indices, sparse_block_size,
                                              sparse_count, batch, n2_idx, s1_idx, cur_actual_seq_lengths_kv)
                mm1_res = np.matmul(q_curr.astype(np.float32), k_sparse.astype(np.float32).T, dtype=matmul_dtype)
                scale_res = mm1_res * scale_value
                
                if sparse_mode == 3:
                    mask_res = mask(scale_res, cur_acutal_seq_lengths_q, cur_actual_seq_lengths_kv,
                                    cur_sparse_indices, s1_idx, sparse_block_size)
                else:
                    mask_res = scale_res
                softmax_res = softmax(mask_res)
                mm2_res = np.matmul(softmax_res, v_sparse, dtype=matmul_dtype)
                y[batch, n2_idx * g: (n2_idx + 1) * g, s1_idx, :] = mm2_res

    if layout_query == 'TND':
        cpu_out = torch.tensor(y)
        return trans_bnsd_to_tnd(cpu_out, cpu_out.shape, actual_seq_lengths_query)
    return np.transpose(y, axes=(0, 2, 1, 3))