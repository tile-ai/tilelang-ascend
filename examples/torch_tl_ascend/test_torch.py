import torch
import torch_npu

import torch_tl_ascend

def ref_flash_attention(q, k, v):
    q = q.float()
    k = k.float()
    v = v.float()

    acc = torch.einsum("bhsd,bhkd->bhsk", q, k) * (1.0 / q.shape[-1])**0.5
    acc = acc.softmax(dim=-1)
    o = torch.einsum("bhsk,bhkd->bhsd", acc, v)
    return o.to(torch.float16)

if __name__ == "__main__":
    B, S, H, D = 4, 4096, 16, 128

    torch.set_default_device('npu')
    torch.manual_seed(0)

    q = torch.randn((B, H, S, D), dtype=torch.float16)
    k = torch.randn((B, H, S, D), dtype=torch.float16)
    v = torch.randn((B, H, S, D), dtype=torch.float16)

    torch.npu.synchronize()
    print("init successful!")

    output = torch.ops.tl_ascend.flash_attention(q, k, v)
    ref_output = ref_flash_attention(q, k, v)
    torch.npu.synchronize()

    torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)

    print("Test Passed!")
