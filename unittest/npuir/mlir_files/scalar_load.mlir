module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf32, #hivm.address_space<gm>>, %arg4: memref<?xf32, #hivm.address_space<gm>>, %arg5: memref<?xf32, #hivm.address_space<gm>>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1024], strides: [%0] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1024], strides: [%0] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1]>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1024], strides: [%0] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1]>, #hivm.address_space<gm>>
    %1 = hivm.hir.get_block_idx -> i64
    %2 = arith.trunci %1 : i64 to i32
    %c1024_i32 = arith.constant 1024 : i32
    %3 = arith.muli %2, %c1024_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %5 = memref.load %reinterpret_cast[%4] : memref<1024xf32, strided<[1]>, #hivm.address_space<gm>>
    %6 = memref.load %reinterpret_cast_1[%4] : memref<1024xf32, strided<[1]>, #hivm.address_space<gm>>
    %7 = arith.addf %5, %6 : f32
    memref.store %7, %reinterpret_cast_0[%4] : memref<1024xf32, strided<[1]>, #hivm.address_space<gm>>
    return
  }
}
