module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c16_i32 = arith.constant 16 : i32
    %1 = arith.muli %c16_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 16], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>
    %3 = hivm.hir.get_block_idx -> i64
    %4 = arith.trunci %3 : i64 to i32
    %alloc = memref.alloc() : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %5 = arith.muli %c0_i32, %c8_i32 : i32
    %6 = arith.addi %5, %4 : i32
    %7 = arith.cmpi slt, %6, %c1_i32 : i32
    scf.if %7 {
      %c0_i32_2 = arith.constant 0 : i32
      %c16_i32_3 = arith.constant 16 : i32
      %8 = arith.muli %6, %c16_i32_3 : i32
      %9 = arith.muli %c0_i32_2, %c16_i32_3 : i32
      %10 = arith.index_cast %8 : i32 to index
      %11 = arith.index_cast %9 : i32 to index
      %subview = memref.subview %reinterpret_cast[%10, %11] [16, 16] [1, 1] : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
      %cst = arith.constant 2.000000e+00 : f16
      %cst_4 = arith.constant -1.000000e+00 : f16
      %cst_5 = arith.constant -1.700000e+01 : f16
      %cst_6 = arith.constant 3.000000e+00 : f16
      %cst_7 = arith.constant 1.500000e+01 : f16
      %cst_8 = arith.constant 3.150000e+02 : f16
      %12 = arith.divf %cst_4, %cst_6 : f16
      %13 = arith.divf %cst, %cst_7 : f16
      %14 = arith.divf %cst_5, %cst_8 : f16
      %alloc_9 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      %alloc_11 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%alloc, %alloc : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>, memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<16x16xf16, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_9, %alloc : memref<16x16xf16, #hivm.address_space<ub>>, memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<16x16xf16, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_10, %alloc_9 : memref<16x16xf16, #hivm.address_space<ub>>, memref<16x16xf16, #hivm.address_space<ub>>) outs(%alloc_11 : memref<16x16xf16, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_11, %alloc_9 : memref<16x16xf16, #hivm.address_space<ub>>, memref<16x16xf16, #hivm.address_space<ub>>) outs(%alloc_12 : memref<16x16xf16, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_10, %12 : memref<16x16xf16, #hivm.address_space<ub>>, f16) outs(%alloc_10 : memref<16x16xf16, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_11, %13 : memref<16x16xf16, #hivm.address_space<ub>>, f16) outs(%alloc_11 : memref<16x16xf16, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_12, %14 : memref<16x16xf16, #hivm.address_space<ub>>, f16) outs(%alloc_12 : memref<16x16xf16, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc, %alloc_10 : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>, memref<16x16xf16, #hivm.address_space<ub>>) outs(%alloc_13 : memref<16x16xf16, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_11, %alloc_13 : memref<16x16xf16, #hivm.address_space<ub>>, memref<16x16xf16, #hivm.address_space<ub>>) outs(%alloc_13 : memref<16x16xf16, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_12, %alloc_13 : memref<16x16xf16, #hivm.address_space<ub>>, memref<16x16xf16, #hivm.address_space<ub>>) outs(%alloc_1 : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>)
      %subview_14 = memref.subview %reinterpret_cast_0[%10, %11] [16, 16] [1, 1] : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %alloc_1, %subview_14 : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>> to memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}
