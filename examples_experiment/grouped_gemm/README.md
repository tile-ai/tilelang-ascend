# TileLang Grouped GEMM for Ascend NPU

This example implements **Grouped Matrix Multiplication (Grouped GEMM)** optimized for Ascend NPUs using [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend).

It is designed to efficiently handle variable-length sequences (stacked into a single tensor `A`) multiplied by unique weight matrices (batched in tensor `B`) for each group.

## 🚀 Key Features

-   **Group-Aligned Tiling**: Tiling strategies respect group boundaries. Computation blocks do not cross between different groups/batches, ensuring isolation and correctness.
-   **Metadata-Driven Indexing**: Uses a pre-computed CPU metadata table to map logical blocks to physical memory addresses, eliminating expensive runtime control flow and logic within the NPU kernel.



## ⚠️ Constraints & Requirements

**Crucial**: The current implementation is a specialized "Aligned Kernel". To ensure maximum performance and avoid hardware memory alignment errors (`ADDR_MISALIGN`), the inputs **must** adhere to the following constraints:

1.  **Batch Size Alignment**:
    *   Every element in `batch_sizes_list` must be **divisible by `block_M`**.
    *   *Reason*: The kernel currently does not handle tail/masked writing. It writes full `block_M` rows back to global memory. Partial blocks will cause memory corruption or crashes.
    *   ✅ Valid: `batch_sizes=[128, 64]`, `block_M=64`
    *   ❌ Invalid: `batch_sizes=[100, 70]`, `block_M=64`

2.  **Dimension Alignment**:
    *   **N** must be **divisible by `block_N`**.
    *   **K** must be **divisible by `block_K`**.


## 🛠 Usage

Run the example script to compile the kernel and verify correctness against a PyTorch reference implementation:

```shell
cd examples/grouped_gemm/
python example_grouped_gemm_fwd.py
```

**Verification:**
If the execution is successful, the script will output:
> Kernel Output Match!

## 📝 TODO

- [ ] Support tail blocks for `batch_sizes` not divisible by `block_M` (Masked Load/Store).
