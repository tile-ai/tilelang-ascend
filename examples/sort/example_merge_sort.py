import tilelang as tl
import tilelang.language as T
import torch

tl.cache.clear_cache()

ELEMENT_SIZE = 2
VALUE_POSITION = 0
INDEX_POSITION = 1

N = 64

pass_configs = {
    tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tl.jit(out_idx=[-1], pass_configs=pass_configs)
def generate_merge_sort_2way():
    @T.prim_func
    def main(
        block0: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block1: T.Tensor([N * ELEMENT_SIZE], "float32"),
        output: T.Tensor([N * 2 * ELEMENT_SIZE], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src0 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src1 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            merge_output = T.alloc_shared([N * 2 * ELEMENT_SIZE], "float32")
            tmp = T.alloc_shared([N * 2 * ELEMENT_SIZE], "float32")

            T.copy(block0, src0)
            T.copy(block1, src1)

            T.tile.merge_sort(merge_output, tmp, src0, src1)

            T.copy(merge_output, output)

    return main


@tl.jit(out_idx=[-1], pass_configs=pass_configs)
def generate_merge_sort_3way():
    @T.prim_func
    def main(
        block0: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block1: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block2: T.Tensor([N * ELEMENT_SIZE], "float32"),
        output: T.Tensor([N * 3 * ELEMENT_SIZE], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src0 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src1 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src2 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            merge_output = T.alloc_shared([N * 3 * ELEMENT_SIZE], "float32")
            tmp = T.alloc_shared([N * 3 * ELEMENT_SIZE], "float32")

            T.copy(block0, src0)
            T.copy(block1, src1)
            T.copy(block2, src2)

            T.tile.merge_sort(merge_output, tmp, src0, src1, src2)

            T.copy(merge_output, output)

    return main


@tl.jit(out_idx=[-1], pass_configs=pass_configs)
def generate_merge_sort_4way():
    @T.prim_func
    def main(
        block0: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block1: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block2: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block3: T.Tensor([N * ELEMENT_SIZE], "float32"),
        output: T.Tensor([N * 4 * ELEMENT_SIZE], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src0 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src1 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src2 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src3 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            merge_output = T.alloc_shared([N * 4 * ELEMENT_SIZE], "float32")
            tmp = T.alloc_shared([N * 4 * ELEMENT_SIZE], "float32")

            T.copy(block0, src0)
            T.copy(block1, src1)
            T.copy(block2, src2)
            T.copy(block3, src3)

            T.tile.merge_sort(merge_output, tmp, src0, src1, src2, src3)

            T.copy(merge_output, output)

    return main


def create_sorted_block(N):
    values = torch.randn(N, dtype=torch.float32)
    sorted_indices = torch.argsort(values, descending=True)
    sorted_values = values[sorted_indices]

    block = torch.zeros(N * ELEMENT_SIZE, dtype=torch.float32)
    for i in range(N):
        block[i * ELEMENT_SIZE + VALUE_POSITION] = sorted_values[i]
        block[i * ELEMENT_SIZE + INDEX_POSITION] = float(sorted_indices[i].item())

    return block


def ref_program(blocks):
    merge_num = len(blocks)
    sequences = []
    for block in blocks:
        pairs = []
        for j in range(N):
            value = block[j * ELEMENT_SIZE + VALUE_POSITION].item()
            index = block[j * ELEMENT_SIZE + INDEX_POSITION].item()
            pairs.append((value, index))
        sequences.append(pairs)

    import heapq

    neg_seqs = [[(-v, i) for v, i in seq] for seq in sequences]
    merged = list(heapq.merge(*neg_seqs))
    merged = [(-v, i) for v, i in merged]

    result = torch.zeros(N * merge_num * ELEMENT_SIZE, dtype=torch.float32)
    for i, (value, index) in enumerate(merged):
        result[i * ELEMENT_SIZE + VALUE_POSITION] = value
        result[i * ELEMENT_SIZE + INDEX_POSITION] = index

    return result


def format_block(block):
    elements = []
    total = len(block) // ELEMENT_SIZE
    for i in range(total):
        value = block[i * ELEMENT_SIZE + VALUE_POSITION].item()
        index = block[i * ELEMENT_SIZE + INDEX_POSITION].item()
        elements.extend([value, index])
    return elements


def test_merge(merge_num):
    print(f"\n{'=' * 60}")
    print(f"Testing {merge_num}-way merge sort (value-index pair format):")
    print(f"N = {N} elements per block, each element = {ELEMENT_SIZE} floats")
    print("=" * 60)

    blocks = [create_sorted_block(N) for _ in range(merge_num)]
    print("blocks", blocks)

    print("\nInput blocks (value, index pairs, all elements):")
    for i in range(merge_num):
        formatted = format_block(blocks[i])
        print(f"  Block {i}: {formatted}")

    blocks_npu = [b.npu() for b in blocks]

    if merge_num == 2:
        kernel = generate_merge_sort_2way()
    elif merge_num == 3:
        kernel = generate_merge_sort_3way()
    elif merge_num == 4:
        kernel = generate_merge_sort_4way()

    print("\nGenerated kernel source:")
    print(kernel.get_kernel_source())

    torch.npu.synchronize()
    print("init successful!")

    result = kernel(*blocks_npu)
    torch.npu.synchronize()

    ref_result = ref_program(blocks)

    result_cpu = result.cpu()
    print(f"\nOutput (all elements): {format_block(result_cpu)}")
    print(f"\nref_result output (all elements): {format_block(ref_result)}")

    output_values = [result_cpu[i * ELEMENT_SIZE + VALUE_POSITION].item() for i in range(N * merge_num)]
    is_sorted = all(output_values[i] >= output_values[i + 1] for i in range(len(output_values) - 1))
    print(f"\nIs output sorted (descending): {is_sorted}")

    ref_values = [ref_result[i * ELEMENT_SIZE + VALUE_POSITION].item() for i in range(N * merge_num)]
    print(f"Match: {output_values == ref_values}")

    if output_values != ref_values:
        correct = sum(1 for i, v in enumerate(output_values) if abs(v - ref_values[i]) < 1e-5)
        print(f"Correct: {correct}/{len(ref_values)}")

    return is_sorted and output_values == ref_values


def main():
    torch.manual_seed(42)

    results = []
    for merge_num in [2, 3, 4]:
        success = test_merge(merge_num)
        results.append((merge_num, success))

    print(f"\n{'=' * 60}")
    print("Summary:")
    for merge_num, success in results:
        status = "Kernel Output Match!" if success else "FAIL"
        print(f"  {merge_num}-way merge: {status}")


if __name__ == "__main__":
    main()
