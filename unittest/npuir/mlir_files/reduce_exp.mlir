module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: memref<?xf16, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
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
    %alloc = memref.alloc() : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
    %alloc_4 = memref.alloc() : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
    %alloc_5 = memref.alloc() : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
    %alloc_6 = memref.alloc() : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
    %alloc_7 = memref.alloc() : memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>>
    %alloc_8 = memref.alloc() : memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>>
    %alloc_9 = memref.alloc() : memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>>
    %alloc_10 = memref.alloc() : memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>>
    %subview = memref.subview %alloc[0, 0] [16, 16] [1, 1] : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
    memref.copy %reinterpret_cast, %subview : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_11 = memref.subview %alloc_4[0, 0] [16, 16] [1, 1] : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
    memref.copy %reinterpret_cast_2, %subview_11 : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_12 = memref.subview %alloc_5[0, 0] [16, 16] [1, 1] : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
    memref.copy %reinterpret_cast_0, %subview_12 : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_13 = memref.subview %alloc_6[0, 0] [16, 16] [1, 1] : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
    memref.copy %reinterpret_cast_3, %subview_13 : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
    hivm.hir.vabs ins(%alloc : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>)
    hivm.hir.vreduce <sum> ins(%alloc : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
    hivm.hir.vreduce <max> ins(%alloc_4 : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
    hivm.hir.vmax ins(%alloc_7, %alloc_8 : memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>>, memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>>)
    hivm.hir.vreduce <min> ins(%alloc_5 : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
    hivm.hir.vmin ins(%alloc_7, %alloc_9 : memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>>, memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>>)
    hivm.hir.vreduce <sum> ins(%alloc_6 : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
    hivm.hir.vadd ins(%alloc_7, %alloc_10 : memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>>, memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>>)
    %subview_14 = memref.subview %reinterpret_cast_1[0, 0] [16, 1] [1, 1] : memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<gm>> to memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<gm>>
    memref.copy %alloc_7, %subview_14 : memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<ub>> to memref<16x1xf16, strided<[1, 1]>, #hivm.address_space<gm>>
    return
  }
}
