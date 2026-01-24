module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c4_i32 = arith.constant 4 : i32
    %1 = arith.muli %c4_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [4, 4], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [4, 4], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<gm>>
    %3 = hivm.hir.get_block_idx -> i64
    %4 = arith.trunci %3 : i64 to i32
    %5 = tensor.empty() : tensor<4x4xf16>
    %6 = tensor.empty() : tensor<4x4xf16>
    %7 = tensor.empty() : tensor<4x4xf16>
    %subview = memref.subview %reinterpret_cast[0, 0] [4, 4] [1, 1] : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<gm>> to memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<4x4xf16>
    %subview_1 = memref.subview %alloc[0, 0] [4, 4] [1, 1] : memref<4x4xf16> to memref<4x4xf16, strided<[4, 1]>>
    memref.copy %subview, %subview_1 : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<gm>> to memref<4x4xf16, strided<[4, 1]>>
    %8 = bufferization.to_tensor %subview_1 restrict : memref<4x4xf16, strided<[4, 1]>>
    %inserted_slice = tensor.insert_slice %8 into %5[0, 0] [4, 4] [1, 1] : tensor<4x4xf16> into tensor<4x4xf16>
    %cst = arith.constant -1.000000e+00 : f16
    %9 = hivm.hir.vmul ins(%inserted_slice, %cst : tensor<4x4xf16>, f16) outs(%7 : tensor<4x4xf16>) -> tensor<4x4xf16>
    %10 = hivm.hir.vexp ins(%9 : tensor<4x4xf16>) outs(%9 : tensor<4x4xf16>) -> tensor<4x4xf16>
    %cst_2 = arith.constant 1.000000e+00 : f16
    %11 = hivm.hir.vadd ins(%10, %cst_2 : tensor<4x4xf16>, f16) outs(%10 : tensor<4x4xf16>) -> tensor<4x4xf16>
    %12 = hivm.hir.vrec ins(%11 : tensor<4x4xf16>) outs(%6 : tensor<4x4xf16>) -> tensor<4x4xf16>
    %extracted_slice = tensor.extract_slice %12[0, 0] [4, 4] [1, 1] : tensor<4x4xf16> to tensor<4x4xf16>
    %subview_3 = memref.subview %reinterpret_cast_0[0, 0] [4, 4] [1, 1] : memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<gm>> to memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<gm>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_3 : (tensor<4x4xf16>, memref<4x4xf16, strided<[4, 1]>, #hivm.address_space<gm>>) -> ()
    return
  }
}
