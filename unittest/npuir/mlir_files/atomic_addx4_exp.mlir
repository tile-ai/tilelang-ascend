module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @atomicAddx4Program(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>>, %arg4: memref<?xf32, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
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
    %c16_i32 = arith.constant 16 : i32
    %5 = arith.divsi %4, %c16_i32 : i32
    %6 = arith.remsi %4, %c16_i32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32_1 = arith.constant 1 : i32
    scf.for %arg13 = %c0_i32 to %c16_i32 step %c1_i32_1  : i32 {
      %c0_i32_2 = arith.constant 0 : i32
      %c4_i32 = arith.constant 4 : i32
      %c1_i32_3 = arith.constant 1 : i32
      scf.for %arg14 = %c0_i32_2 to %c4_i32 step %c1_i32_3  : i32 {
        %7 = tensor.empty() : tensor<1x4xf32>
        %c16_i32_4 = arith.constant 16 : i32
        %8 = arith.muli %5, %c16_i32_4 : i32
        %9 = arith.addi %8, %arg13 : i32
        %10 = arith.muli %6, %c16_i32_4 : i32
        %c4_i32_5 = arith.constant 4 : i32
        %11 = arith.muli %arg14, %c4_i32_5 : i32
        %12 = arith.addi %10, %11 : i32
        %13 = arith.index_cast %9 : i32 to index
        %14 = arith.index_cast %12 : i32 to index
        %subview = memref.subview %reinterpret_cast[%13, %14] [1, 4] [1, 1] : memref<256x256xf32, strided<[256, 1]>, #hivm.address_space<gm>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %alloc = memref.alloc() : memref<4xf32>
        %subview_6 = memref.subview %alloc[0] [4] [1] : memref<4xf32> to memref<4xf32, strided<[1]>>
        memref.copy %subview, %subview_6 : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<4xf32, strided<[1]>>
        %15 = bufferization.to_tensor %subview_6 restrict : memref<4xf32, strided<[1]>>
        %inserted_slice = tensor.insert_slice %15 into %7[0, 0] [1, 4] [1, 1] : tensor<4xf32> into tensor<1x4xf32>
        %subview_7 = memref.subview %reinterpret_cast_0[%13, %14] [1, 4] [1, 1] : memref<256x256xf32, strided<[256, 1]>, #hivm.address_space<gm>> to memref<1x4xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.store ins(%inserted_slice : tensor<1x4xf32>) outs(%subview_7 : memref<1x4xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) atomic = <add>
      }
    }
    return
  }
}
