module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: memref<?xf16, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c16_i32 = arith.constant 16 : i32
    %1 = arith.muli %c16_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [16, 16], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>
    %3 = arith.muli %c1_i32, %c1_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [16, 1], strides: [%4, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 16], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [16, 16], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>
    %5 = hivm.hir.get_block_idx -> i64
    %6 = arith.trunci %5 : i64 to i32
    %7 = tensor.empty() : tensor<16x16xf16>
    %8 = tensor.empty() : tensor<16x16xf16>
    %9 = tensor.empty() : tensor<16x16xf16>
    %10 = tensor.empty() : tensor<16x16xf16>
    %11 = tensor.empty() : tensor<16x1xf16>
    %12 = tensor.empty() : tensor<16x1xf16>
    %13 = tensor.empty() : tensor<16x1xf16>
    %14 = tensor.empty() : tensor<16x1xf16>
    %subview = memref.subview %reinterpret_cast[0, 0] [16, 16] [1, 1] : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<16x16xf16>
    %subview_4 = memref.subview %alloc[0, 0] [16, 16] [1, 1] : memref<16x16xf16> to memref<16x16xf16, strided<[16, 1]>>
    memref.copy %subview, %subview_4 : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>>
    %15 = bufferization.to_tensor %subview_4 restrict : memref<16x16xf16, strided<[16, 1]>>
    %inserted_slice = tensor.insert_slice %15 into %7[0, 0] [16, 16] [1, 1] : tensor<16x16xf16> into tensor<16x16xf16>
    %subview_5 = memref.subview %reinterpret_cast_2[0, 0] [16, 16] [1, 1] : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>
    %alloc_6 = memref.alloc() : memref<16x16xf16>
    %subview_7 = memref.subview %alloc_6[0, 0] [16, 16] [1, 1] : memref<16x16xf16> to memref<16x16xf16, strided<[16, 1]>>
    memref.copy %subview_5, %subview_7 : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>>
    %16 = bufferization.to_tensor %subview_7 restrict : memref<16x16xf16, strided<[16, 1]>>
    %inserted_slice_8 = tensor.insert_slice %16 into %8[0, 0] [16, 16] [1, 1] : tensor<16x16xf16> into tensor<16x16xf16>
    %subview_9 = memref.subview %reinterpret_cast_0[0, 0] [16, 16] [1, 1] : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>
    %alloc_10 = memref.alloc() : memref<16x16xf16>
    %subview_11 = memref.subview %alloc_10[0, 0] [16, 16] [1, 1] : memref<16x16xf16> to memref<16x16xf16, strided<[16, 1]>>
    memref.copy %subview_9, %subview_11 : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>>
    %17 = bufferization.to_tensor %subview_11 restrict : memref<16x16xf16, strided<[16, 1]>>
    %inserted_slice_12 = tensor.insert_slice %17 into %9[0, 0] [16, 16] [1, 1] : tensor<16x16xf16> into tensor<16x16xf16>
    %subview_13 = memref.subview %reinterpret_cast_3[0, 0] [16, 16] [1, 1] : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>
    %alloc_14 = memref.alloc() : memref<16x16xf16>
    %subview_15 = memref.subview %alloc_14[0, 0] [16, 16] [1, 1] : memref<16x16xf16> to memref<16x16xf16, strided<[16, 1]>>
    memref.copy %subview_13, %subview_15 : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>>
    %18 = bufferization.to_tensor %subview_15 restrict : memref<16x16xf16, strided<[16, 1]>>
    %inserted_slice_16 = tensor.insert_slice %18 into %10[0, 0] [16, 16] [1, 1] : tensor<16x16xf16> into tensor<16x16xf16>
    %19 = hivm.hir.vabs ins(%inserted_slice : tensor<16x16xf16>) outs(%inserted_slice : tensor<16x16xf16>) -> tensor<16x16xf16>
    %20 = hivm.hir.vreduce <sum> ins(%19 : tensor<16x16xf16>) outs(%11 : tensor<16x1xf16>) reduce_dims = [1] -> tensor<16x1xf16>
    %21 = hivm.hir.vreduce <max> ins(%inserted_slice_8 : tensor<16x16xf16>) outs(%12 : tensor<16x1xf16>) reduce_dims = [1] -> tensor<16x1xf16>
    %22 = hivm.hir.vmax ins(%20, %21 : tensor<16x1xf16>, tensor<16x1xf16>) outs(%20 : tensor<16x1xf16>) -> tensor<16x1xf16>
    %23 = hivm.hir.vreduce <min> ins(%inserted_slice_12 : tensor<16x16xf16>) outs(%13 : tensor<16x1xf16>) reduce_dims = [1] -> tensor<16x1xf16>
    %24 = hivm.hir.vmin ins(%22, %23 : tensor<16x1xf16>, tensor<16x1xf16>) outs(%22 : tensor<16x1xf16>) -> tensor<16x1xf16>
    %25 = hivm.hir.vreduce <sum> ins(%inserted_slice_16 : tensor<16x16xf16>) outs(%14 : tensor<16x1xf16>) reduce_dims = [1] -> tensor<16x1xf16>
    %26 = hivm.hir.vadd ins(%24, %25 : tensor<16x1xf16>, tensor<16x1xf16>) outs(%24 : tensor<16x1xf16>) -> tensor<16x1xf16>
    %extracted_slice = tensor.extract_slice %26[0, 0] [16, 1] [1, 1] : tensor<16x1xf16> to tensor<16x1xf16>
    %subview_17 = memref.subview %reinterpret_cast_1[0, 0] [16, 1] [1, 1] : memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<gm>> to memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<gm>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_17 : (tensor<16x1xf16>, memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<gm>>) -> ()
    return
  }
}
