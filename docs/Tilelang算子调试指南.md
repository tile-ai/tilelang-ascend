# 1 引言

本文档为​**Tilelang-AscendNPUIR 调试指南**​，面向参与 Tilelang与昇腾（Ascend）NPU 适配开发的工程师，系统性地介绍在Tilelang到AscendNPUIR 编译与运行过程中常用的调试方法与工具。
全文内容概览如下：

| 章节  | 主要内容  |
| ------------ | ------------ | 
| **1.概述** | 说明Tilelang算子调试中的常见问题分类和调试指引 |
| **2.编译流程概览**| 说明Tilelang-AscendNPUIR端到端编译全流程，方便用户定位问题调试目标 | 
| **3.调试方法** | 基于编译流程介绍每个阶段的调试方法 |

## 1.1 Tilelang算子调试常见问题

在开发过程中，问题通常可归纳为以下几类。下表提供了快速的问题类型辨识与首选调试方法指引。

| 问题类型  | 具体表现  | 推荐调试方法  |
| ------------ | ------------ |------------ |
| **1.精度问题** | 算子成功编译并生成.o文件，但NPU运行结果和标杆参考（如torch）存在差异 |  `T.print`打印调试 |
| **2.编译失败**| 算子编译失败，未生成预期的TVM IR和MLIR | 编译调试 |
| **3.运行时失败** | 算子编译成功，但未生成.o文件，进程终止 | 运行时调试 |

# 2 Tilelang-AscendNPUIR 编译流程概览

| 编译阶段  | 输入  | 输出  | 工具/组件  | 说明 |
| ------------ | ------------ |------------ |------------ |------------ |
| **Python Kernel编译** | Tilelang_kernel.py | TVM IR  | TVM编译器  | 将用户编写的前端DSL语言（Tilelang Kernel）编译为TVM IR |
| **Codegen阶段**| TVM IR | MLIR  | JIT + Tilelang Codegen  | 将TVM IR转换为面向Ascend NPU后端的MLIR |
| **AscendNPUIR编译** | MLIR | .o（可执行文件）  | JIT + 毕昇编译器（bishengir-compile）  | 将MLIR进一步编译并优化，生成可在NPU上执行的二进制代码 |

Tilelang算子的完整调用流程示意如下：

```
[Python Tilelang] ->(前端DSL语言）
	↓  调用tilelang/language/init中的算子接口，返回TIR调用节点，经过lower中TVM对应pass  ←调试阶段1
[TVM IR]
	↓  执行codegen，生成AscendNPUIR的输入mlir ←调试阶段2
[MLIR]
	↓  执行AscendNPUIR的PASS (bishengir-compile) ←调试阶段3
[NPU 可执行文件.o]
	↓  Runtime
[计算结果输出]
```

# 3 Tilelang-AscendNPUIR 编译中间IR解读

## 3.1 TVM IR

TVM IR 是 TVM 编译器前端生成的中间表示（Intermediate Representation），在语义上和前端DSL语言较为相似，保留了原始 Tilelang Python 内核的语义结构。

```
@I.ir_module
class Module:
    @T.prim_func
    def main(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle, M: T.int32):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "stackvm", "tag": ""}, "keys": [], "kind": "npuir", "tag": ""})})
        A = T.match_buffer(A_handle, (M, 64), "float16")
        B = T.match_buffer(B_handle, (M, 64), "float16")
        C = T.match_buffer(C_handle, (M, 64), "float16")
        cid = T.launch_thread("blockIdx.x", M // 32)
        _ = T.launch_thread("blockIdx.y", M // 32 * 2)
        A_VEC = T.decl_buffer((32, 64), "float16", scope="shared")
        B_VEC = T.decl_buffer((32, 64), "float16", scope="shared")
        C_VEC = T.decl_buffer((32, 64), "float16", scope="shared")
        T.copy(T.region(A[cid * 32, 0], 1, 32, 64), T.region(A_VEC[0, 0], 2, 32, 64))
        T.copy(T.region(B[cid * 32, 0], 1, 32, 64), T.region(B_VEC[0, 0], 2, 32, 64))
        for i in range(32):
            T.npuir_add(T.region(A_VEC[i, 0], 1, 1, 1), T.region(B_VEC[i, 0], 1, 1, 1), T.region(C_VEC[i, 0], 2, 1, 1))
        T.copy(T.region(C_VEC[0, 0], 1, 32, 1), T.region(C[cid * 32, 0], 2, 32, 1))

```

TVM IR解析要点

- 数据类型：`T.decl_buffer`表示在TVM中声明buffer内存，`scope="shared.dyn"`表示buffer类型，这里shared对应NPU架构中的UB内存，其他相关scope类型可以在tilelang/language/allocate.py中查询；`T.region`表示TVM中的一块内存切片，包括起始位置和偏移，在customize_npuir.py中有将buffer和其他TVM数据类型转换为region的映射函数；`T.region(A[cid * 32, 0], 1, 32, 64)`这里[cid * 32, 0]表示起始位置，1表示access_type，1对应read，2对应write; 32表示第一维的偏移，64表示第二维的偏移。
- 操作类op：在tilelang/_init.py中可以找到不同类别的op注册信息和对应的注册位置，例如`T.npuir_add`在`customize_npuir.py`中，`alloc_ub`在`allocate.py`中。
- `T.serial` 本身表示串行循环，在IR中展开为循环结构

## 3.2 MLIR（Tilelang IR）

### 3.2.1 Develop模式

Develop模式下的mlir样例如下

```
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16>, %arg4: memref<?xf16>, %arg5: memref<?xf16>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg6 : i32 to index
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.index_cast %c1_i32 : i32 to index
    %c32_i32 = arith.constant 32 : i32
    %2 = arith.muli %c32_i32, %c1_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, 32], strides: [%3, %1] : memref<?xf16> to memref<?x64xf16, strided<[?, ?]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, 32], strides: [%3, %1] : memref<?xf16> to memref<?x64xf16, strided<[?, ?]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, 32], strides: [%3, %1] : memref<?xf16> to memref<?x64xf16, strided<[?, ?]>>
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.trunci %4 : i64 to i32
    %6 = tensor.empty() : tensor<32x64xf16>
    %7 = tensor.empty() : tensor<32x64xf16>
    %8 = tensor.empty() : tensor<32x64xf16>
    %9 = arith.muli %5, %c32_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %subview = memref.subview %reinterpret_cast[%10, 0] [32, 32] [1, 1] : memref<?x64xf16, strided<[?, ?]>> to memref<32x64xf16, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() : memref<32x64xf16>
    memref.copy %subview, %alloc : memref<32x64xf16, strided<[?, ?], offset: ?>> to memref<32x64xf16>
    %11 = bufferization.to_tensor %alloc restrict : memref<32x64xf16>
    %inserted_slice = tensor.insert_slice %11 into %6[0, 0] [32, 32] [1, 1] : tensor<32x64xf16> into tensor<32x64xf16>
    %subview_2 = memref.subview %reinterpret_cast_1[%10, 0] [32, 32] [1, 1] : memref<?x64xf16, strided<[?, ?]>> to memref<32x64xf16, strided<[?, ?], offset: ?>>
    %alloc_3 = memref.alloc() : memref<32x64xf16>
    memref.copy %subview_2, %alloc_3 : memref<32x64xf16, strided<[?, ?], offset: ?>> to memref<32x64xf16>
    %12 = bufferization.to_tensor %alloc_3 restrict : memref<32x64xf16>
    %inserted_slice_4 = tensor.insert_slice %12 into %7[0, 0] [32, 32] [1, 1] : tensor<32x64xf16> into tensor<32x64xf16>
    %13 = hivm.hir.vadd ins(%inserted_slice, %inserted_slice_4 : tensor<32x64xf16>, tensor<32x64xf16>) outs(%8 : tensor<32x64xf16>) -> tensor<32x64xf16>
    %extracted_slice = tensor.extract_slice %13[0, 0] [32, 1] [1, 1] : tensor<32x64xf16> to tensor<32x1xf16>
    %subview_5 = memref.subview %reinterpret_cast_0[%10, 0] [32, 1] [1, 1] : memref<?x64xf16, strided<[?, ?]>> to memref<32x1xf16, strided<[?, ?], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_5 : (tensor<32x1xf16>, memref<32x1xf16, strided<[?, ?], offset: ?>>) -> ()
    return
  }
}

```

TilelangIR 是将TVM IR转换为适配昇腾 NPU 架构的中间表示（Intermediate Representation），采用标准 MLIR dialect（如 `memref`、`linalg`、`scf` 等）

- `memref.alloc` & `memref.copy`: 显式内存层级管理；%alloc = memref.alloc() : memref<32x64xf16>表示在快速存储区（如 UB 或 Shared Memory）分配一块固定大小的临时缓冲区
- `memref.reinterpret_cast`: 动态形状与布局重构；处理动态输入形状（Symbolic Shapes），例如%reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, 32], strides: [%3, %1] : memref<?xf16> to memref<?x64xf16, strided<[?, ?]>>表示把一个未知shape的变量%arg3映射成<?x64xf16>的shape;
- `memref.subview`: 数据切片；从`memref` 中截取一个固定大小的子区域（Sub-region）;
- `bufferization.to_tensor`: 内存到计算的转换； `memref` 中的数据视图转换为 `tensor` 类型，供后续计算单元（Vector / Cube）使用
- `tensor.insert_slice` & `tensor.extract_slice`: 对张量进行插入与截取；例如%extracted_slice = tensor.extract_slice %13[0, 0] [32, 1] [1, 1] : tensor<32x64xf16> to tensor<32x1xf16>表示从`%13`中截取一个<32x1xf16>的张量；
- `bufferization.materialize_in_destination`: 零拷贝写回；将`tensor` （写入）到指定的 `memref` 内存地址中；

### 3.2.2 Expert模式

expert模式生成的mlir和dev类似，核心区别是dev上使用tensor语义，在AscendNPUIR中进一步转换为memref，而expert中直接codegen成mlir；例如在dev中的`%6 = tensor.empty() : tensor<32x32xf16>` 对应在expert中是`%alloc = memref.alloc() : memref<32x32xf16, strided<[32, 1]>, #hivm.address_space<ub>>`

# 4 调试方法

下文按照调用的4个阶段介绍一些通用或常见的调试场景和方法。

## 4.1 从TVM IR到运行时的调试方法（常用在编译失败）

### 4.1.1 DSL->TVM IR层--对应tilelang/language(python前端)

从DSL到TVM IR的转换通常较为直观，对代码原有逻辑的保留较为明显，可以通过TVM IR检查某些运算是否符合预期，例如Op操作数据类型、大小，Op功能是否生效，某些Op是否被映射为正确的TIR调用节点。
此外，在tilelang前端到TVM IR转换中，主要经过`tilelang-ascend/tilelang/engine/lower.py`中的前两个阶段

```
Phase 1: Lower and legalize the IR
Phase 2: Optimize the IR for the target
```

其中在phase2，npuir分支中执行了包括`NpuLoopVectorize`（对ParallelOp进行向量化改写）在内的优化pass，对其的调试也发生在这一阶段。

- 示例一：判断Op功能是否生效，以T.parallel为例

```python
for i, j in T.Parallel(block_M, n):
      A[i, j] = T.sigmoid(B[i, j] * C[0] + D[0, j]) + eps
# 这段代码对应的正确的TVM IR是
↓
T.npuir_mul(T.region(B[0, 0], 1, 64, 4), T.region(C[0], 1, 1), T.region(tmp_0_buf[0, 0], 2, 64, 4))
T.npuir_add(T.region(tmp_0_buf[0, 0], 1, 64, 4), T.region(D[0, 0], 1, 1, 4), T.region(tmp_1_buf[0, 0], 2, 64, 4))
T.npuir_sigmoid(T.region(tmp_1_buf[0, 0], 1, 64, 4), T.region(tmp_2_buf[0, 0], 2, 64, 4))
T.npuir_add(T.region(tmp_2_buf[0, 0], 1, 64, 4), T.float32(9.9999999999999995e-07), T.region(A[0, 0], 2, 64, 4))
```

但当parallel没有支持T.sigmoidOp时，parallel无法进行向量化，得到的TVM IR如下

```
for i in T.serial(block_M):
	for j in T.serial(n) 
      		A[i, j] = T.sigmoid(B[i, j] * C[0] + D[0, j]) + eps
```

- 示例二：检查Op的数据类型是否正确，以T.vadd为例

```
T.npuir_add(A_VEC, eps, B_VEC)
↓ 正确的结果应该是
T.npuir_add(T.region(A_VEC[0, 0], 1, 32, 32), T.float32(9.9999999999999995e-07), T.region(B_VEC[0, 0], 1, 32, 32))
```

- 示例三：检查某些Op是否被正确映射，以T.clear为例 T.clearOp被映射为T.brcOp，这个映射逻辑定义在tilelang-ascend/tilelang/language/customize_npuir.py中，npuir_clear函数实际返回的是`npuir_fill(buffer, zero)`接口，如果TVM IR不符合预期，则可以修改customize_npuir.py中的定义来进行修改

```
T.clear(A_VEC)
↓
T.npuir_brc(0, T.region(A_VEC[0, 0], 2, 128, 256))
```

**常见的报错场景**
场景一：TVM层的报错，通常会给出具体的报错位置

```
error: 'float' object has no attribute 'buffer'
 --> /home/d00957057/tilelang-ascend/testing/npuir/test_vec_add_2d_scalar.py:39:21
    |  
 39 |                      T.npuir_add(A_VEC, eps, C_VEC)
    |                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
note: run with `TVM_BACKTRACE=1` environment variable to display a backtrace.

```

- **调试方法：在tilelang/language中找到对应op的python定义接口，使用pdb调试**
  调试示例如下，在`tilelang/language/customize_npuir.py`中定位到npuir_add接口，进一步定位到AscendBinaryOp，插入断点，当程序暂停后，打印检查输入参数src1的数据类型，发现是float而不是tvm支持的常量数据类型

```python
def npuir_add(A, B, C):
    return AscendBinaryOp("add", A, B, C).buildTirCall()
class AscendBinaryOp(object):
       """
       Args:
           A (Union[tir.Buffer, tir.Var]): Input argument to legalize
           B (Union[tir.Buffer, tir.Var]): Input argument to legalize
           C (Union[tir.Buffer, tir.Var]): Output argument to legalize
       Returns:
           tir.Call: A handle to the npuir binary operation
       """
       def __init__(self, opName, src0, src1, dst):
           self.__opName = opName
           self.__src0 = src0
           self.__src1 = src1
           self.__dst = dst
       def buildTirCall(self):

           src0 = _to_region(self.__src0, "r", _get_extent(self.__src0))
           import pdb
           pdb.set_trace()
		   src1 = _to_region(self.__src1, "r", _get_extent(self.__src1))
		     ↓ 修改如下，增加判断条件，如果是int/float，则改为TVM支持的tir.const类型
           src1 = tir.const(self.__src1, self.__dst.dtype) if isinstance(self.__src1,(int,float)) else _to_region(self.__src1, "r", _get_extent(self.__src1))
           dst = _to_region(self.__dst, "w", _get_extent(self.__dst))
           return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_" + self.__opName), src0, src1, dst)

(Pdb) p dir(self) # 检查输入参数有哪些可以查看的属性 ->找到src1对应 '_AscendBinaryOp__src1'
(Pdb) p type(self._AscendBinaryOp__src1)  # 检查输入参数src1的数据类型 ->输出<class 'float'>
(Pdb) n  # 单步执行到下一行
(Pdb) p src1 -> # T.float16(0.001) 
(Pdb) p type(src1) -># <class 'tvm.tir.expr.FloatImm'>  数据类型修改成功，从float修改为tvm.tir.expr.FloatImm

```

---

### 4.1.2 TVM IR->MLIR层--对应codegen

TVM IR通过codegen生成MLIR，如果TVM IR成功生成，但是没有生成NPUIR对应的mlir，说明codegen中的代码存在问题。
如果缺少具体的报错信息不完整（只显示segmentation fault core dump），可以将生成的MLIR复制成.mlir文件，并运行**bishengir-opt .mlir**，得到具体的报错信息

示例一： 没有发生core dump, 但是报错信息不具体

```
File "/home/d00957057/tilelang-ascend/tilelang/jit/jit_npu.py", line 1154, in _parse_npuir_metadata
    kernel_name = re.search(KERNEL_NAME_REGEX, self.mlir_content).group(1)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'group'
```

将生成的mlir放入单独文件，运行bishengir-opt debug.mlir，得到具体报错如下，可知是add输入形状不一致

```
debug.mlir:59:9: error: 'hivm.hir.vadd' op operands' shape are inconsistent
        "hivm.hir.vadd"(%22, %48, %24) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>, memref<1x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>) -> ()
```

如果想深入codegen代码打印中间运算结果，可以进一步调试。
**调试方法：在`tilelang-ascend/src/target` 中找到当前模式对应的codegen.cpp文件（例如dev模式对应的`codegen_npuir_dev.cc`）通过调试语句或GDB进行调试**
**调试语句示例**：在`CodeGenTileLangNPUIRDEV::VreduceCodegen`中插入debug语句，判断`T.reduce(A, row_max, dims=2, reduce_mode="max", size=[block_M, 4, 4])`中的size切片是否生效

```C++
void CodeGenTileLangNPUIRDEV::VreduceCodegen(const CallNode *op) {
  tvm::tl::NpuirReduce npuirop(op->args, this->vmap);
  Value src = GetVarValue(npuirop.src);
  Value dst = GetVarValue(npuirop.dst);
// 插入debug语句
  llvm::dbgs() << "[DEBUG] VreduceCodegen, src=" << src.getType() << "\n";
```

修改codegen代码之后， 需要在`tilelang-ascend/build`路径下执行make重新进行编译，再次运行得到debug语句如下，发现src的shape是<64x4x8xf32>而不是期望的<64x4x4xf32>，说明切片未生效，分析发现 Value src = GetVarValue(npuirop.src)这里使用的是原始Op而不是切片，修改.cpp文件，从src中提取切片，Value src = GenExtractSliceFromRegion(npuirop.src, npuirop.src_range);重新编译运行后得到正确结果。

```
[DEBUG] VreduceCodegen, src=tensor<64x4x8xf32>
↓ 修改后的输出
[DEBUG] VreduceCodegen, src=tensor<64x4x4xf32>
```

示例二：没有生成报错信息，直接显示`Segmentation fault (core dumped)`
**GDB调试**：当无法推测可能引起core dump的代码位置时，需要用GDB深入跟踪C++代码执行流

- 用GDB定位core dump位置，之后可进一步调试codegen代码进行修改。
  
  ```C++
  //执行下面的gdb 语句
  gdb --args python .py
  (gdb) r
  //打印具体报错
  Thread 1 "python" received signal SIGSEGV, Segmentation fault.
   0x00007ffeec825606 in tvm::runtime::Array<tvm::PrimExpr, void>::operator[](long) const [clone .constprop.0] () from /tilelang-ascend/build/libtilelang_module.so ->//TVM编译器在尝试解析或优化 TileLang 代码时，内部维护的一个数组（Array）发生了越界访问或访问了空指针
  ```

---

### 4.1.3 MLIR->.o文件--对应AscendNPUIR PASS

如果TVM IR和MLIR都成功编译，但是没有编出.o文件，说明在AscendNPUIR PASS存在失败的pass，此时需要打印完整的Pass来定位。
首先在install_npuir.sh文件中增加`--enable-ir-print`这个编译选项，然后重新编译，并配置环境变量为新编译的bishengir-compile路径

```
if [ -z "$BISHENGIR_PATH" ]; then
    echo "warring: no --bishengir-path set, bishengir path will be found in environment variable PATH"
    # build bishengir in 3rdparty
    echo "build bishengir in 3rdparty"
    git submodule update --init --recursive 3rdparty/AscendNPU-IR
    pushd 3rdparty/AscendNPU-IR
    bash ./build-tools/apply_patches.sh
    rm -rf ./build
    ./build-tools/build.sh -o ./build --build-torch-mlir --c-compiler=clang --cxx-compiler=clang++ \
    --add-cmake-options="-DCMAKE_LINKER=lld -DLLVM_ENABLE_LLD=ON" --enable-ir-print --apply-patches --bishengir-publish=off
    BISHENGIR_PATH="./3rdparty/AscendNPU-IR/build/install"
    popd
fi
```

运行如下指令打印完整Pass

```
/tilelang-ascend-dev/3rdparty/AscendNPU-IR/build/bin/bishengir-compile \
  tilelang_debug.mlir \
  --enable-auto-multi-buffer=True \
  --enable-auto-bind-sub-block=True \
  --enable-hfusion-compile=true \
  --enable-hivm-compile=true \
  --enable-triton-kernel-compile=true \
  -o test.o \
  --mlir-print-ir-before-all \
  --mlir-print-ir-after-all \
  --mlir-disable-threading \
  >compile.log 2>&1

```

在全量pass中，可以搜索fail定位失败的pass，一类常见的报错是UB overflow，当发生UB overflow，PlanMemory pass会失败。UB overflow通常是由于在ub分配了过多的内存，一个有效的调试方法是修改block的大小，减小单次循环的block大小。例如，下面发生UB overflow的算子中block=32，当把block修改为16，报错消失。

```
A = T.alloc_shared((block, dim), dtype)
# 当block = 32 出现如下报错
↓
loc("/tmp/tmpa48n0m0b/kernel.npuir":2:3): error: UB overflow, requires 1579008 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)
```

上面的报错有明确的报错信息，但在有的场景中，并不会显示具体的报错，例如

```
err cmd: /home/CANN/CANN8.5.0/cann-8.5.0/bin/bishengir-compile /tmp/tmp_fyvndkz/kernel.npuir --enable-auto-multi-buffer=false --enable-triton-kernel-compile=true --enable-hivm-compile=true --limit-auto-multi-buffer-only-for-local-buffer=true --enable-auto-bind-sub-block=true -o /tmp/tmp_fyvndkz/kernel
err code: -11
err info: PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:

```

此时，打印全量Pass，发现编译在PlanMemory的下一个Pass失败，比较PlanMemory的Before和After发现PlanMemory并未生效，可进一步深入PlanMemory 逻辑分析原因。

---

---

## 4.2 打印调试：T.print 打印中间结果（常用在精度定位）

成功生成.o文件后，算子运行并得到输出结果，如果能正常输出，但是精度错误，可以通过T.print打印中间计算结果来分析具体是哪个步骤运算结果不符合预期。
T.print用法如下：其中obj表示要打印的变量（var或buffer），msg表示可选的提示信息，hex表示结果是否以16进制打印（默认false）

```
T.print(obj, msg, hex)
```

下面是一个调试示例

```
T.reduce(A, row_sum, dims=1, reduce_mode="sum", size=[4, 4], clear = True)
T.reduce(A, row_sum_2, dims=1, reduce_mode="sum", size=[4, 8], clear = True)
T.print(row_sum)
T.print(row_sum_2)
```

发现打印结果row_sum和row_sum2完全相同，不符合预期，说明size没有生效，进一步定位reduce的codegen代码发现reduce在识别src时识别的是原始op而不是切片，修改代码逻辑进行修正。

---

## 4.3 在jit_npu中增加编译选项进行调试（常用在性能优化）

在jit_npu中找到compile_option_list，其中有3个默认编译选项"--enable-auto-multi-buffer=true" 表示开始自动multi-buffer选项，此外，"--limit-auto-multi-buffer-only-for-local-buffer=false"表示开启核间流水， "--enable-auto-bind-sub-block=true"表示开启自动进行C:V 1:2分核。可根据需要开启合适的编译选项。

```
_compile_option_list = [
           "--enable-auto-multi-buffer=false",
           "--enable-triton-kernel-compile=true",
           "--enable-hivm-compile=true",
            ]
```

# 5 总结

本文介绍了Tilelang-AscendNPUIR的完整编译流程、常见的调试问题和对应的调试方法。

- **精度问题**：通过T.print打印来对比中间计算结果，定位算子计算错误原因。
- **编译问题**：基于多个编译阶段分层定位，通过pdb和gdb等debug方法定位失败原因，查看IR来辅助理解算子功能是否符合预期。
- **运行时失败**：xx。
- **性能优化**：通过控制jit_npu中的编译选项来配置流水优化等功能。

掌握这些调试技巧后，您就可以在TileLang中高效地调试NPU算子，充分利用昇腾硬件的计算能力。