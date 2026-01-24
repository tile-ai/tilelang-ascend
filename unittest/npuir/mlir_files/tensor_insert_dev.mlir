module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @insert(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>>, %arg4: memref<?xf32, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c4_i32 = arith.constant 4 : i32
    %1 = arith.muli %c4_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [4, 4], strides: [%2, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<4x4xf32, strided<[4, 1]>, #hivm.address_space<gm>>
    %c16_i32 = arith.constant 16 : i32
    %3 = arith.muli %c16_i32, %c1_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 16], strides: [%4, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<1x16xf32, strided<[16, 1]>, #hivm.address_space<gm>>
    %5 = hivm.hir.get_block_idx -> i64
    %6 = arith.trunci %5 : i64 to i32
    %7 = tensor.empty() : tensor<4x4xf32>
    %8 = tensor.empty() : tensor<1x16xf32>
    %subview = memref.subview %reinterpret_cast_0[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[16, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<16xf32>
    %subview_1 = memref.subview %alloc[0] [16] [1] : memref<16xf32> to memref<16xf32, strided<[1]>>
    memref.copy %subview, %subview_1 : memref<16xf32, strided<[1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
    %9 = bufferization.to_tensor %subview_1 restrict : memref<16xf32, strided<[1]>>
    %inserted_slice = tensor.insert_slice %9 into %8[0, 0] [1, 16] [1, 1] : tensor<16xf32> into tensor<1x16xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32_2 = arith.constant 1 : i32
    %10 = scf.for %arg11 = %c0_i32 to %c4_i32 step %c1_i32_2 iter_args(%arg12 = %7) -> (tensor<4x4xf32>)  : i32 {
      %c0_i32_4 = arith.constant 0 : i32
      %c4_i32_5 = arith.constant 4 : i32
      %c1_i32_6 = arith.constant 1 : i32
      %11 = scf.for %arg13 = %c0_i32_4 to %c4_i32_5 step %c1_i32_6 iter_args(%arg14 = %arg12) -> (tensor<4x4xf32>)  : i32 {
        %c0_i32_7 = arith.constant 0 : i32
        %12 = arith.index_cast %c0_i32_7 : i32 to index
        %c4_i32_8 = arith.constant 4 : i32
        %13 = arith.muli %arg11, %c4_i32_8 : i32
        %14 = arith.addi %13, %arg13 : i32
        %15 = arith.index_cast %14 : i32 to index
        %extracted = tensor.extract %inserted_slice[%12, %15] : tensor<1x16xf32>
        %16 = arith.index_cast %arg11 : i32 to index
        %17 = arith.index_cast %arg13 : i32 to index
        %inserted = tensor.insert %extracted into %arg14[%16, %17] : tensor<4x4xf32>
        scf.yield %inserted : tensor<4x4xf32>
      }
      scf.yield %11 : tensor<4x4xf32>
    }
    %extracted_slice = tensor.extract_slice %10[0, 0] [4, 4] [1, 1] : tensor<4x4xf32> to tensor<4x4xf32>
    %subview_3 = memref.subview %reinterpret_cast[0, 0] [4, 4] [1, 1] : memref<4x4xf32, strided<[4, 1]>, #hivm.address_space<gm>> to memref<4x4xf32, strided<[4, 1]>, #hivm.address_space<gm>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_3 : (tensor<4x4xf32>, memref<4x4xf32, strided<[4, 1]>, #hivm.address_space<gm>>) -> ()
    return
  }
}
