module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf32, #hivm.address_space<gm>>, %arg4: memref<?xf32, #hivm.address_space<gm>>, %arg5: memref<?xf32, #hivm.address_space<gm>>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c64_i32 = arith.constant 64 : i32
    %1 = arith.muli %c64_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64, 64], strides: [%2, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [64, 64], strides: [%2, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<gm>>
    %c128_i32 = arith.constant 128 : i32
    %3 = arith.muli %c128_i32, %c1_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [128, 128], strides: [%4, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<gm>>
    %5 = hivm.hir.get_block_idx -> i64
    %6 = arith.trunci %5 : i64 to i32
    %alloc = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_3 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %c2_i32 = arith.constant 2 : i32
    %7 = arith.divsi %6, %c2_i32 : i32
    %c16_i32 = arith.constant 16 : i32
    %8 = arith.muli %7, %c16_i32 : i32
    %9 = arith.remsi %6, %c2_i32 : i32
    %c32_i32 = arith.constant 32 : i32
    %10 = arith.muli %9, %c32_i32 : i32
    %11 = arith.index_cast %8 : i32 to index
    %12 = arith.index_cast %10 : i32 to index
    %subview = memref.subview %reinterpret_cast[%11, %12] [16, 32] [1, 1] : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<gm>> to memref<16x32xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc : memref<16x32xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %cst = arith.constant 0.000000e+00 : f32
    hivm.hir.vpad ins(%alloc : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_2 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) low[8, 0] high[8, 0] pad_value %cst : f32
    %13 = arith.index_cast %6 : i32 to index
    hivm.hir.vpad ins(%alloc : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_3 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) low[%13, 0] high[%13, 0] pad_value %cst : f32
    %14 = arith.muli %c2_i32, %8 : i32
    %15 = arith.index_cast %14 : i32 to index
    %subview_4 = memref.subview %reinterpret_cast_1[%15, %12] [32, 32] [1, 1] : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<gm>> to memref<32x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %alloc_2, %subview_4 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_5 = memref.subview %reinterpret_cast_0[%11, %12] [32, 32] [1, 1] : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<gm>> to memref<32x32xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %alloc_3, %subview_5 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32x32xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    return
  }
}
