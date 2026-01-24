module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %c256_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [256, 256], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<256x256xf16, strided<[256, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [256, 256], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<256x256xf16, strided<[256, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [256, 256], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<256x256xf16, strided<[256, 1]>, #hivm.address_space<gm>>
    %3 = hivm.hir.get_block_idx -> i64
    %4 = arith.trunci %3 : i64 to i32
    %alloc = memref.alloc() : memref<32x64xi1, strided<[64, 1]>, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<32x64xf16, strided<[64, 1]>, #hivm.address_space<ub>>
    %alloc_3 = memref.alloc() : memref<32x64xf16, strided<[64, 1]>, #hivm.address_space<ub>>
    %alloc_4 = memref.alloc() : memref<32x64xf16, strided<[64, 1]>, #hivm.address_space<ub>>
    %c4_i32 = arith.constant 4 : i32
    %5 = arith.divsi %4, %c4_i32 : i32
    %6 = arith.remsi %4, %c4_i32 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = arith.muli %5, %c32_i32 : i32
    %8 = arith.index_cast %7 : i32 to index
    %c64_i32 = arith.constant 64 : i32
    %9 = arith.muli %6, %c64_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %subview = memref.subview %reinterpret_cast[%8, %10] [32, 64] [1, 1] : memref<256x256xf16, strided<[256, 1]>, #hivm.address_space<gm>> to memref<32x64xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc_2 : memref<32x64xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x64xf16, strided<[64, 1]>, #hivm.address_space<ub>>
    %subview_5 = memref.subview %reinterpret_cast_1[%8, %10] [32, 64] [1, 1] : memref<256x256xf16, strided<[256, 1]>, #hivm.address_space<gm>> to memref<32x64xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_5, %alloc_3 : memref<32x64xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x64xf16, strided<[64, 1]>, #hivm.address_space<ub>>
    hivm.hir.vcmp ins(%alloc_2, %alloc_3 : memref<32x64xf16, strided<[64, 1]>, #hivm.address_space<ub>>, memref<32x64xf16, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<32x64xi1, strided<[64, 1]>, #hivm.address_space<ub>>) compare_mode = <ge>
    %subview_6 = memref.subview %alloc[0, 0] [32, 64] [1, 1] : memref<32x64xi1, strided<[64, 1]>, #hivm.address_space<ub>> to memref<32x64xi1, strided<[64, 1]>, #hivm.address_space<ub>>
    hivm.hir.vsel ins(%subview_6, %alloc_2, %alloc_3 : memref<32x64xi1, strided<[64, 1]>, #hivm.address_space<ub>>, memref<32x64xf16, strided<[64, 1]>, #hivm.address_space<ub>>, memref<32x64xf16, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<32x64xf16, strided<[64, 1]>, #hivm.address_space<ub>>)
    %subview_7 = memref.subview %reinterpret_cast_0[%8, %10] [32, 64] [1, 1] : memref<256x256xf16, strided<[256, 1]>, #hivm.address_space<gm>> to memref<32x64xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %alloc_4, %subview_7 : memref<32x64xf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<32x64xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    return
  }
}
