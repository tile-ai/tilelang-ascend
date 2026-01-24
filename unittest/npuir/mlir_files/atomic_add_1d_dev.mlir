module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @vecAtomicAdd1D(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>>, %arg4: memref<?xf32, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64], strides: [%0] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [64], strides: [%0] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %1 = hivm.hir.get_block_idx -> i64
    %2 = arith.trunci %1 : i64 to i32
    %3 = tensor.empty() : tensor<32xf32>
    %c32_i32 = arith.constant 32 : i32
    %4 = arith.muli %2, %c32_i32 : i32
    %5 = arith.subi %arg5, %4 : i32
    %6 = arith.minsi %c32_i32, %5 : i32
    %7 = arith.index_cast %4 : i32 to index
    %8 = arith.index_cast %6 : i32 to index
    %subview = memref.subview %reinterpret_cast[%7] [%8] [1] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<32xf32>
    %subview_1 = memref.subview %alloc[0] [%8] [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_1 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1]>>
    %9 = bufferization.to_tensor %subview_1 restrict : memref<?xf32, strided<[1]>>
    %inserted_slice = tensor.insert_slice %9 into %3[0] [%8] [1] : tensor<?xf32> into tensor<32xf32>
    %subview_2 = memref.subview %reinterpret_cast_0[%7] [%8] [1] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%inserted_slice : tensor<32xf32>) outs(%subview_2 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) atomic = <add>
    return
  }
}
