# tilelang.jit.compile

Compile the given TileLang PrimFunc and construct JITKernel_NPU.

## Function signature

```
tilelang.jit.compile(
    func=None,
    out_idx=None,
    execution_backend="cython",
    target="auto",
    target_host=None,
    verbose=False,
    pass_configs=None,
)
```

## Parameters

* **func** (*tvm.tir.PrimFunc* )
* **out_idx** (​*Union[List[int], int, None]* )
* **execution_backend** (*Literal['dlpack', 'ctypes', 'cython', 'nvrtc']* )
* **target** (*"npuir"* )
* **target_host** (*Union[str, tvm.target.Target, None]* )
* **verbose** (*bool* )
* **pass_configs** (*dict, optional* )
* **compile_flags** (*Optional[Union[List[str], str]]* )

<span style="color: red;">Note</span>: If `target="npuir"` is not manually specified, the CUDA runtime will be used.

## Return Type

JITKernel_NPU

## Usage Example

```
# Basic Usage - The target="npuir" must be specified. 
 kernel = tilelang.jit.compile(func=my_func, target="npuir")

# Full example
 kernel = tilelang.jit.compile(
     func=my_prim_func,
     out_idx=0,
     execution_backend="cython",
     target="npuir",  # 必须设置为 "npuir"
     verbose=True,
     pass_configs={
         "tir.disable_vectorize": True,
         "tl.dynamic_vectorize_size_bits": 256
     }
 )
```
