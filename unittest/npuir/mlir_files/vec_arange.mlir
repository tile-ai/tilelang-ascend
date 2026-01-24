module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c512_i32 = arith.constant 512 : i32
    %1 = arith.muli %c512_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [512, 512], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [512, 512], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>>
    %3 = hivm.hir.get_block_idx -> i64
    %4 = arith.trunci %3 : i64 to i32
    %alloc = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
    %c2_i32 = arith.constant 2 : i32
    %5 = arith.divsi %4, %c2_i32 : i32
    %c128_i32 = arith.constant 128 : i32
    %6 = arith.muli %5, %c128_i32 : i32
    %7 = arith.remsi %4, %c2_i32 : i32
    %c256_i32 = arith.constant 256 : i32
    %8 = arith.muli %7, %c256_i32 : i32
    %c1_i64 = arith.constant 1 : i64
    %9 = arith.index_cast %c1_i64 : i64 to index
    %c1_i64_2 = arith.constant 1 : i64
    %10 = arith.index_cast %c1_i64_2 : i64 to index
    %c2_i64 = arith.constant 2 : i64
    %11 = arith.index_cast %c2_i64 : i64 to index
    hivm.hir.varange offset[%9] strides[%10, %11] outs(%alloc : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
    %c0_i64 = arith.constant 0 : i64
    %12 = arith.index_cast %c0_i64 : i64 to index
    %c1_i64_3 = arith.constant 1 : i64
    %13 = arith.index_cast %c1_i64_3 : i64 to index
    %c2_i64_4 = arith.constant 2 : i64
    %14 = arith.index_cast %c2_i64_4 : i64 to index
    hivm.hir.varange offset[%12] strides[%13, %14] outs(%alloc_1 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
    %15 = arith.index_cast %6 : i32 to index
    %16 = arith.index_cast %8 : i32 to index
    %subview = memref.subview %reinterpret_cast[%15, %16] [128, 256] [1, 1] : memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>> to memref<128x256xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %alloc, %subview : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<128x256xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_5 = memref.subview %reinterpret_cast_0[%15, %16] [128, 256] [1, 1] : memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>> to memref<128x256xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %alloc_1, %subview_5 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<128x256xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    return
  }
}
