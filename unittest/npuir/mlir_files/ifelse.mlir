module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c1024_i32 = arith.constant 1024 : i32
    %1 = arith.muli %c1024_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1024, 1024], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<1024x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1024, 1024], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<1024x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1024, 1024], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<1024x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1024, 1024], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<1024x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>>
    %3 = hivm.hir.get_block_idx -> i64
    %4 = arith.trunci %3 : i64 to i32
    %c0_i32 = arith.constant 0 : i32
    %c7_i32 = arith.constant 7 : i32
    %c1_i32_3 = arith.constant 1 : i32
    scf.for %arg13 = %c0_i32 to %c7_i32 step %c1_i32_3  : i32 {
      %alloc = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
      %alloc_4 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
      %alloc_5 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
      %c10_i32 = arith.constant 10 : i32
      %5 = arith.muli %arg13, %c10_i32 : i32
      %6 = arith.remsi %4, %c10_i32 : i32
      %7 = arith.addi %5, %6 : i32
      %c64_i32 = arith.constant 64 : i32
      %8 = arith.cmpi slt, %7, %c64_i32 : i32
      scf.if %8 {
        %c4_i32 = arith.constant 4 : i32
        %9 = arith.divsi %7, %c4_i32 : i32
        %10 = arith.remsi %7, %c4_i32 : i32
        %c64_i32_6 = arith.constant 64 : i32
        %11 = arith.muli %9, %c64_i32_6 : i32
        %c256_i32 = arith.constant 256 : i32
        %12 = arith.muli %10, %c256_i32 : i32
        %13 = arith.index_cast %11 : i32 to index
        %14 = arith.index_cast %12 : i32 to index
        %subview = memref.subview %reinterpret_cast[%13, %14] [64, 256] [1, 1] : memref<1024x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>> to memref<64x256xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview, %alloc : memref<64x256xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
        %subview_7 = memref.subview %reinterpret_cast_1[%13, %14] [64, 256] [1, 1] : memref<1024x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>> to memref<64x256xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_7, %alloc_4 : memref<64x256xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
        %c10_i32_8 = arith.constant 10 : i32
        %15 = arith.cmpi slt, %4, %c10_i32_8 : i32
        scf.if %15 {
          hivm.hir.vadd ins(%alloc, %alloc_4 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
        } else {
          hivm.hir.vsub ins(%alloc, %alloc_4 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
        }
        scf.if %15 {
          %16 = arith.index_cast %11 : i32 to index
          %17 = arith.index_cast %12 : i32 to index
          %subview_9 = memref.subview %reinterpret_cast_0[%16, %17] [64, 256] [1, 1] : memref<1024x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>> to memref<64x256xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_5, %subview_9 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<64x256xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        } else {
          %16 = arith.index_cast %11 : i32 to index
          %17 = arith.index_cast %12 : i32 to index
          %subview_9 = memref.subview %reinterpret_cast_2[%16, %17] [64, 256] [1, 1] : memref<1024x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>> to memref<64x256xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_5, %subview_9 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<64x256xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        }
      }
    }
    return
  }
}
