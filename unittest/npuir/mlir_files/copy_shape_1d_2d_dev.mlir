module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @copyShape(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c1024_i32 = arith.constant 1024 : i32
    %1 = arith.muli %c1024_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [256, 1024], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<256x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [256, 1024], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<256x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>>
    %3 = hivm.hir.get_block_idx -> i64
    %4 = arith.trunci %3 : i64 to i32
    %c32_i32 = arith.constant 32 : i32
    %5 = arith.divsi %4, %c32_i32 : i32
    %6 = arith.remsi %4, %c32_i32 : i32
    %7 = arith.muli %6, %c32_i32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32_1 = arith.constant 1 : i32
    scf.for %arg13 = %c0_i32 to %c32_i32 step %c1_i32_1  : i32 {
      %8 = tensor.empty() : tensor<32xf16>
      %c32_i32_2 = arith.constant 32 : i32
      %9 = arith.muli %5, %c32_i32_2 : i32
      %10 = arith.addi %9, %arg13 : i32
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %7 : i32 to index
      %subview = memref.subview %reinterpret_cast[%11, %12] [1, 32] [1, 1] : memref<256x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>> to memref<32xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %alloc = memref.alloc() : memref<32xf16>
      %subview_3 = memref.subview %alloc[0] [32] [1] : memref<32xf16> to memref<32xf16, strided<[1]>>
      memref.copy %subview, %subview_3 : memref<32xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<32xf16, strided<[1]>>
      %13 = bufferization.to_tensor %subview_3 restrict : memref<32xf16, strided<[1]>>
      %inserted_slice = tensor.insert_slice %13 into %8[0] [32] [1] : tensor<32xf16> into tensor<32xf16>
      %extracted_slice = tensor.extract_slice %inserted_slice[0] [32] [1] : tensor<32xf16> to tensor<32xf16>
      %subview_4 = memref.subview %reinterpret_cast_0[%11, %12] [1, 32] [1, 1] : memref<256x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>> to memref<1x32xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %from_elements = tensor.from_elements %c1, %c32 : tensor<2xindex>
      %reshape = tensor.reshape %extracted_slice(%from_elements) : (tensor<32xf16>, tensor<2xindex>) -> tensor<1x32xf16>
      bufferization.materialize_in_destination %reshape in writable %subview_4 : (tensor<1x32xf16>, memref<1x32xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
    }
    return
  }
}
