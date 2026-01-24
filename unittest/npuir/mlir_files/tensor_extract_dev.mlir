module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @add(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>>, %arg4: memref<?xf32, #hivm.address_space<gm>>, %arg5: memref<?xf32, #hivm.address_space<gm>>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c4_i32 = arith.constant 4 : i32
    %1 = arith.muli %c4_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [4, 4], strides: [%2, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<4x4xf32, strided<[4, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [4, 4], strides: [%2, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<4x4xf32, strided<[4, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [3], strides: [%0] : memref<?xf32, #hivm.address_space<gm>> to memref<3xf32, strided<[1]>, #hivm.address_space<gm>>
    %3 = hivm.hir.get_block_idx -> i64
    %4 = arith.trunci %3 : i64 to i32
    %5 = tensor.empty() : tensor<1x4xf32>
    %6 = tensor.empty() : tensor<3xf32>
    %7 = tensor.empty() : tensor<1x4xf32>
    %8 = arith.index_cast %4 : i32 to index
    %subview = memref.subview %reinterpret_cast[%8, 0] [1, 4] [1, 1] : memref<4x4xf32, strided<[4, 1]>, #hivm.address_space<gm>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<4xf32>
    %subview_2 = memref.subview %alloc[0] [4] [1] : memref<4xf32> to memref<4xf32, strided<[1]>>
    memref.copy %subview, %subview_2 : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<4xf32, strided<[1]>>
    %9 = bufferization.to_tensor %subview_2 restrict : memref<4xf32, strided<[1]>>
    %inserted_slice = tensor.insert_slice %9 into %5[0, 0] [1, 4] [1, 1] : tensor<4xf32> into tensor<1x4xf32>
    %subview_3 = memref.subview %reinterpret_cast_1[0] [3] [1] : memref<3xf32, strided<[1]>, #hivm.address_space<gm>> to memref<3xf32, strided<[1]>, #hivm.address_space<gm>>
    %alloc_4 = memref.alloc() : memref<3xf32>
    %subview_5 = memref.subview %alloc_4[0] [3] [1] : memref<3xf32> to memref<3xf32, strided<[1]>>
    memref.copy %subview_3, %subview_5 : memref<3xf32, strided<[1]>, #hivm.address_space<gm>> to memref<3xf32, strided<[1]>>
    %10 = bufferization.to_tensor %subview_5 restrict : memref<3xf32, strided<[1]>>
    %inserted_slice_6 = tensor.insert_slice %10 into %6[0] [3] [1] : tensor<3xf32> into tensor<3xf32>
    %c0_i32 = arith.constant 0 : i32
    %11 = arith.index_cast %c0_i32 : i32 to index
    %extracted = tensor.extract %inserted_slice_6[%11] : tensor<3xf32>
    %12 = hivm.hir.vadd ins(%inserted_slice, %extracted : tensor<1x4xf32>, f32) outs(%7 : tensor<1x4xf32>) -> tensor<1x4xf32>
    %extracted_slice = tensor.extract_slice %12[0, 0] [1, 4] [1, 1] : tensor<1x4xf32> to tensor<1x4xf32>
    %subview_7 = memref.subview %reinterpret_cast_0[%8, 0] [1, 4] [1, 1] : memref<4x4xf32, strided<[4, 1]>, #hivm.address_space<gm>> to memref<1x4xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_7 : (tensor<1x4xf32>, memref<1x4xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
    return
  }
}
