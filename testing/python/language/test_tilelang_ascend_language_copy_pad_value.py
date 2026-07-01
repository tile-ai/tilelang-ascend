import pytest

import tilelang
import tilelang.language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


def copy_pad_value_kernel(valid_cols=63, gm_cols=64, ub_cols=64, pad_value=None, explicit_pad_value=False):
    @T.prim_func
    def main(
        A: T.Tensor((1, gm_cols), "float16"),
        B: T.Tensor((1, gm_cols), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((1, ub_cols), "float16")

            with T.Scope("V"):
                if explicit_pad_value:
                    T.copy(A[0, 0:valid_cols], a_ub[0, 0:valid_cols], pad_value=pad_value)
                else:
                    T.copy(A[0, 0:valid_cols], a_ub[0, 0:valid_cols])
                T.copy(a_ub[0, 0:valid_cols], B[0, 0:valid_cols])

    return main


def compile_pto_source(program):
    func = tilelang.compile(
        program,
        out_idx=[-1],
        pass_configs=pass_configs,
        target="pto",
    )
    return func.get_kernel_source()


def gm_to_ub_copy_lines(code):
    return [line.strip() for line in code.splitlines() if "copy_gm_to_ub_dynamic<" in line or "copy_gm_to_ub_dynamic(" in line]


def test_sliced_default_copy_pad_value_generates_null_for_pto():
    code = compile_pto_source(copy_pad_value_kernel())
    copy_lines = gm_to_ub_copy_lines(code)

    assert copy_lines
    assert any("pto::PadValue::Null" in line for line in copy_lines)
    assert not any("pto::PadValue::Zero" in line for line in copy_lines)


def test_explicit_zero_copy_pad_value_generates_zero_for_pto():
    code = compile_pto_source(copy_pad_value_kernel(valid_cols=64, gm_cols=64, ub_cols=64, pad_value=0, explicit_pad_value=True))
    copy_lines = gm_to_ub_copy_lines(code)

    assert copy_lines
    assert any("pto::PadValue::Zero" in line for line in copy_lines)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
