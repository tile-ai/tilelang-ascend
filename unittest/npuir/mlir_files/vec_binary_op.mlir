module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c512_i32 = arith.constant 512 : i32
    %1 = arith.muli %c512_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [512, 512], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [512, 512], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [512, 512], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>>
    %3 = hivm.hir.get_block_idx -> i64
    %4 = arith.trunci %3 : i64 to i32
    %alloc = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
    %alloc_3 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
    %c0_i32 = arith.constant 0 : i32
    %c20_i32 = arith.constant 20 : i32
    %5 = arith.muli %c0_i32, %c20_i32 : i32
    %6 = arith.addi %5, %4 : i32
    %c2_i32 = arith.constant 2 : i32
    %7 = arith.divsi %6, %c2_i32 : i32
    %8 = arith.remsi %6, %c2_i32 : i32
    %c128_i32 = arith.constant 128 : i32
    %9 = arith.muli %7, %c128_i32 : i32
    %c256_i32 = arith.constant 256 : i32
    %10 = arith.muli %8, %c256_i32 : i32
    %11 = arith.index_cast %9 : i32 to index
    %12 = arith.index_cast %10 : i32 to index
    %subview = memref.subview %reinterpret_cast[%11, %12] [128, 256] [1, 1] : memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>> to memref<128x256xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc : memref<128x256xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
    %subview_4 = memref.subview %reinterpret_cast_1[%11, %12] [128, 256] [1, 1] : memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>> to memref<128x256xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_4, %alloc_2 : memref<128x256xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.vor ins(%alloc, %alloc_2 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>, memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_3 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
    hivm.hir.vand ins(%alloc, %alloc_2 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>, memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_3 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
    %subview_5 = memref.subview %reinterpret_cast_0[%11, %12] [128, 256] [1, 1] : memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>> to memref<128x256xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %alloc_3, %subview_5 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<128x256xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    return
  }
}
