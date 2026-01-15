import torch
import torch_npu

torch.set_default_device('npu')
torch.manual_seed(0)

B, S, H, D = 4, 4096, 32, 512

q = torch.randn(B, H, S, D, dtype=torch.float16)
k = torch.randn(B, H, S, D, dtype=torch.float16)
v = torch.randn(B, H, S, D, dtype=torch.float16)
print("init successful!")

sm_scale = (1.0 / D)**0.5
output = torch_npu.npu_fusion_attention(
  q, k, v, H,
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
    q = q.float()
    k = k.float()
    v = v.float()

    acc = torch.einsum("bhsd,bhkd->bhsk", q, k) * (1.0 / q.shape[-1])**0.5
    acc = acc.softmax(dim=-1)
    o = torch.einsum("bhsk,bhkd->bhsd", acc, v)
    return o.to(torch.float16)

ref_output = ref_flash_attn(q, k, v)
torch.npu.synchronize()

torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
print("Test Passed!")