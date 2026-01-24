module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf32, #hivm.address_space<gm>>, %arg4: memref<?xi32, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1024], strides: [%0] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1024], strides: [%0] : memref<?xi32, #hivm.address_space<gm>> to memref<1024xi32, strided<[1]>, #hivm.address_space<gm>>
    %1 = hivm.hir.get_block_idx -> i64
    %2 = arith.trunci %1 : i64 to i32
    %cst = arith.constant 1.001000e+02 : f32
    %cst_1 = arith.constant 2.002000e+02 : f32
    %3 = arith.addf %cst, %cst_1 : f32
    %4 = arith.subf %cst, %cst_1 : f32
    %5 = arith.addf %3, %4 : f32
    %6 = arith.subf %4, %3 : f32
    %7 = arith.mulf %5, %6 : f32
    %8 = arith.divf %5, %6 : f32
    %9 = arith.addf %7, %8 : f32
    %cst_2 = arith.constant 7.770000e+00 : f32
    %10 = arith.addf %9, %cst_2 : f32
    %c1024_i32 = arith.constant 1024 : i32
    %11 = arith.muli %2, %c1024_i32 : i32
    %12 = arith.index_cast %11 : i32 to index
    memref.store %10, %reinterpret_cast[%12] : memref<1024xf32, strided<[1]>, #hivm.address_space<gm>>
    %c100_i32 = arith.constant 100 : i32
    %c200_i32 = arith.constant 200 : i32
    %13 = arith.addi %c100_i32, %c200_i32 : i32
    %14 = arith.subi %c100_i32, %c200_i32 : i32
    %15 = arith.addi %13, %14 : i32
    %16 = arith.subi %14, %13 : i32
    %17 = arith.muli %15, %16 : i32
    %18 = arith.divsi %15, %16 : i32
    %19 = arith.addi %17, %18 : i32
    %20 = arith.subi %19, %c1_i32 : i32
    %21 = arith.divsi %20, %18 : i32
    memref.store %21, %reinterpret_cast_0[%12] : memref<1024xi32, strided<[1]>, #hivm.address_space<gm>>
    return
  }
}
