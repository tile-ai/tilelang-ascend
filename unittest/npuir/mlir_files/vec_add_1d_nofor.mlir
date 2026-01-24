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
    %alloc = memref.alloc() : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
    %alloc_3 = memref.alloc() : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
    %c1024_i32 = arith.constant 1024 : i32
    %3 = arith.muli %2, %c1024_i32 : i32
    %4 = arith.subi %arg6, %3 : i32
    %5 = arith.minsi %c1024_i32, %4 : i32
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %5 : i32 to index
    %subview = memref.subview %reinterpret_cast[%6] [%7] [1] : memref<1024xf32, strided<[1]>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %alloc[0] [%7] [1] : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    memref.copy %subview, %subview_4 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_5 = memref.subview %reinterpret_cast_1[%6] [%7] [1] : memref<1024xf32, strided<[1]>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_6 = memref.subview %alloc_2[0] [%7] [1] : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    memref.copy %subview_5, %subview_6 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc, %alloc_2 : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>, memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%alloc_3 : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_7 = memref.subview %alloc_3[0] [%7] [1] : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_8 = memref.subview %reinterpret_cast_0[%6] [%7] [1] : memref<1024xf32, strided<[1]>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_7, %subview_8 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    return
  }
}
