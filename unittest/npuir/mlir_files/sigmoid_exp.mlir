module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c4_i32 = arith.constant 4 : i32
    %1 = arith.muli %c4_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [4, 4], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [4, 4], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<gm>>
    %3 = hivm.hir.get_block_idx -> i64
    %4 = arith.trunci %3 : i64 to i32
    %alloc = memref.alloc() : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<ub>>
    %subview = memref.subview %alloc[0, 0] [4, 4] [1, 1] : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<ub>> to memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<ub>>
    memref.copy %reinterpret_cast, %subview : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<gm>> to memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<ub>>
    %cst = arith.constant -1.000000e+00 : f16
    hivm.hir.vmul ins(%alloc, %cst : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<ub>>, f16) outs(%alloc_2 : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<ub>>)
    hivm.hir.vexp ins(%alloc_2 : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<ub>>) outs(%alloc_2 : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<ub>>)
    %cst_3 = arith.constant 1.000000e+00 : f16
    hivm.hir.vadd ins(%alloc_2, %cst_3 : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<ub>>, f16) outs(%alloc_2 : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<ub>>)
    hivm.hir.vrec ins(%alloc_2 : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<ub>>) outs(%alloc_1 : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<ub>>)
    %subview_4 = memref.subview %reinterpret_cast_0[0, 0] [4, 4] [1, 1] : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<gm>> to memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<gm>>
    memref.copy %alloc_1, %subview_4 : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<ub>> to memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<gm>>
    return
  }
}
