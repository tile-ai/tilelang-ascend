import argparse
import torch
import torch_npu

torch.set_default_device('npu')
torch.manual_seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=4, help="batch size")
    parser.add_argument("--S", type=int, default=4096, help="seq len")
    parser.add_argument("--H", type=int, default=32, help="attention head size")
    parser.add_argument("--q-heads", type=int, default=None, help="query head size")
    parser.add_argument("--kv-heads", type=int, default=None, help="kv head size")
    parser.add_argument("--D", type=int, default=512, help="hidden dim")
    parser.add_argument("--no-check", action="store_true", help="disable reference check")
    args = parser.parse_args()
    B, S, H, D = args.B, args.S, args.H, args.D
    Q_H = args.q_heads or H
    KV_H = args.kv_heads or H

    q = torch.randn(B, Q_H, S, D, dtype=torch.float16)
    k = torch.randn(B, KV_H, S, D, dtype=torch.float16)
    v = torch.randn(B, KV_H, S, D, dtype=torch.float16)
    print("init successful!")

    sm_scale = (1.0 / D)**0.5
    output = torch_npu.npu_fusion_attention(
      q, k, v, Q_H,
      padding_mask=None,
      atten_mask=None,
      scale=sm_scale,
      keep_prob=1.0,
      input_layout="BNSD",
      pre_tockens=65535,
      next_tockens=65535,
      sparse_mode=0,
    )[0]
    torch.npu.synchronize()

    def ref_flash_attn(q, k, v):
        # GQA/MQA support: torch.einsum does not support MQA/GQA broadcasting, so we must manually repeat k/v heads
        if k.shape[1] != q.shape[1]:
            n_rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        q = q.float()
        k = k.float()
        v = v.float()

        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )
        return output.to(torch.float16)

    if not args.no_check:
        ref_output = ref_flash_attn(q, k, v)
        torch.npu.synchronize()
        torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
        print("Test Passed!")
    else:
        print("Reference check skipped.")
