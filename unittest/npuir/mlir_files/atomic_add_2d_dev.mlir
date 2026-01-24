module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @vecAtomicAdd2D(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>>, %arg4: memref<?xf32, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %c256_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [256, 256], strides: [%2, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<256x256xf32, strided<[256, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [256, 256], strides: [%2, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<256x256xf32, strided<[256, 1]>, #hivm.address_space<gm>>
    %3 = hivm.hir.get_block_idx -> i64
    %4 = arith.trunci %3 : i64 to i32
    %5 = tensor.empty() : tensor<16x16xf32>
    %c16_i32 = arith.constant 16 : i32
    %6 = arith.divsi %4, %c16_i32 : i32
    %7 = arith.muli %6, %c16_i32 : i32
    %8 = arith.remsi %4, %c16_i32 : i32
    %9 = arith.muli %8, %c16_i32 : i32
    %10 = arith.subi %arg5, %7 : i32
    %11 = arith.minsi %c16_i32, %10 : i32
    %12 = arith.subi %arg6, %9 : i32
    %13 = arith.minsi %c16_i32, %12 : i32
    %14 = arith.index_cast %7 : i32 to index
    %15 = arith.index_cast %11 : i32 to index
    %16 = arith.index_cast %9 : i32 to index
    %17 = arith.index_cast %13 : i32 to index
    %subview = memref.subview %reinterpret_cast[%14, %16] [%15, %17] [1, 1] : memref<256x256xf32, strided<[256, 1]>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<16x16xf32>
    %subview_1 = memref.subview %alloc[0, 0] [%15, %17] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
    memref.copy %subview, %subview_1 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[16, 1]>>
    %18 = bufferization.to_tensor %subview_1 restrict : memref<?x?xf32, strided<[16, 1]>>
    %inserted_slice = tensor.insert_slice %18 into %5[0, 0] [%15, %17] [1, 1] : tensor<?x?xf32> into tensor<16x16xf32>
    %subview_2 = memref.subview %reinterpret_cast_0[%14, %16] [%15, %17] [1, 1] : memref<256x256xf32, strided<[256, 1]>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%inserted_slice : tensor<16x16xf32>) outs(%subview_2 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) atomic = <add>
    return
  }
}
