module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf32, #hivm.address_space<gm>>, %arg4: memref<?xi32, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1024], strides: [%0] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1024], strides: [%0] : memref<?xi32, #hivm.address_space<gm>> to memref<1024xi32, strided<[1]>, #hivm.address_space<gm>>
    %1 = hivm.hir.get_block_idx -> i64
    %2 = arith.trunci %1 : i64 to i32
    %cst = arith.constant 1.000000e-10 : f32
    %cst_1 = arith.constant 2.000000e+02 : f32
    %3 = arith.addf %cst, %cst_1 : f32
    %c1024_i32 = arith.constant 1024 : i32
    %4 = arith.muli %2, %c1024_i32 : i32
    %5 = arith.index_cast %4 : i32 to index
    memref.store %3, %reinterpret_cast[%5] : memref<1024xf32, strided<[1]>, #hivm.address_space<gm>>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %6 = arith.addi %4, %c1_i32 : i32
    %7 = arith.index_cast %6 : i32 to index
    memref.store %cst_2, %reinterpret_cast[%7] : memref<1024xf32, strided<[1]>, #hivm.address_space<gm>>
    %c100_i32 = arith.constant 100 : i32
    %c200_i32 = arith.constant 200 : i32
    %8 = arith.addi %c100_i32, %c200_i32 : i32
    memref.store %8, %reinterpret_cast_0[%5] : memref<1024xi32, strided<[1]>, #hivm.address_space<gm>>
    return
  }
}
