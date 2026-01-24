module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c512_i32 = arith.constant 512 : i32
    %1 = arith.muli %c512_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [512, 512], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>>
    %3 = hivm.hir.get_block_idx -> i64
    %4 = arith.trunci %3 : i64 to i32
    %alloc = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
    %c2_i32 = arith.constant 2 : i32
    %5 = arith.divsi %4, %c2_i32 : i32
    %c128_i32 = arith.constant 128 : i32
    %6 = arith.muli %5, %c128_i32 : i32
    %7 = arith.remsi %4, %c2_i32 : i32
    %c256_i32 = arith.constant 256 : i32
    %8 = arith.muli %7, %c256_i32 : i32
    %9 = arith.index_cast %6 : i32 to index
    %10 = arith.index_cast %8 : i32 to index
    %subview = memref.subview %reinterpret_cast[%9, %10] [128, 256] [1, 1] : memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>> to memref<128x256xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc : memref<128x256xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
    %11 = hivm.hir.bitcast %alloc : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> -> memref<128x256xi16, strided<[256, 1]>, #hivm.address_space<ub>>
    return
  }
}
