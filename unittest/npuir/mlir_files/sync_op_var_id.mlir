module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @main_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c512_i32 = arith.constant 512 : i32
    %1 = arith.muli %c512_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1024, 512], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<1024x512xf16, strided<[512, 1]>, #hivm.address_space<gm>>
    %c1024_i32 = arith.constant 1024 : i32
    %3 = arith.muli %c1024_i32, %c1_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1024, 1024], strides: [%4, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<1024x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [512, 1024], strides: [%4, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<512x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1024, 1024], strides: [%4, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<1024x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>>
    %5 = hivm.hir.get_block_idx -> i64
    %6 = arith.trunci %5 : i64 to i32
    %7 = hivm.hir.get_sub_block_idx -> i64
    %8 = arith.trunci %7 : i64 to i32
    %alloc = memref.alloc() : memref<128x512xf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
    %alloc_3 = memref.alloc() : memref<512x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
    %alloc_4 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
    %c4_i32 = arith.constant 4 : i32
    %9 = arith.divsi %6, %c4_i32 : i32
    %c128_i32 = arith.constant 128 : i32
    %10 = arith.muli %9, %c128_i32 : i32
    %11 = arith.remsi %6, %c4_i32 : i32
    %c256_i32 = arith.constant 256 : i32
    %12 = arith.muli %11, %c256_i32 : i32
    %13 = arith.subi %arg7, %10 : i32
    %14 = arith.minsi %c128_i32, %13 : i32
    %15 = arith.subi %arg8, %12 : i32
    %16 = arith.minsi %c256_i32, %15 : i32
    %17 = arith.index_cast %10 : i32 to index
    %18 = arith.index_cast %14 : i32 to index
    %subview = memref.subview %reinterpret_cast[%17, 0] [%18, 512] [1, 1] : memref<1024x512xf16, strided<[512, 1]>, #hivm.address_space<gm>> to memref<?x512xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_5 = memref.subview %alloc[0, 0] [%18, 512] [1, 1] : memref<128x512xf16, strided<[512, 1]>, #hivm.address_space<cbuf>> to memref<?x512xf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x512xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?x512xf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
    %19 = arith.index_cast %12 : i32 to index
    %20 = arith.index_cast %16 : i32 to index
    %subview_6 = memref.subview %reinterpret_cast_1[0, %19] [512, %20] [1, 1] : memref<512x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>> to memref<512x?xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_7 = memref.subview %alloc_3[0, 0] [512, %20] [1, 1] : memref<512x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>> to memref<512x?xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%subview_6 : memref<512x?xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_7 : memref<512x?xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
    %true = arith.constant true
    %21 = arith.index_cast %c512_i32 : i32 to index
    hivm.hir.mmadL1 ins(%alloc, %alloc_3, %true, %18, %21, %20 : memref<128x512xf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, memref<512x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_4 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
    %c2_i32 = arith.constant 2 : i32
    %22 = arith.remsi %6, %c2_i32 : i32
    %c5_i32 = arith.constant 5 : i32
    %23 = arith.addi %22, %c5_i32 : i32
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_FIX>] flag = 1
    %subview_8 = memref.subview %alloc_4[0, 0] [%18, %20] [1, 1] : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<cc>>
    %subview_9 = memref.subview %reinterpret_cast_0[%17, %19] [%18, %20] [1, 1] : memref<1024x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%subview_8 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%subview_9 : memref<?x?xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>)
    %24 = arith.extsi %23 : i32 to i64
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %24 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
    %c0_i32 = arith.constant 0 : i32
    %c15_i32 = arith.constant 15 : i32
    %c1_i32_10 = arith.constant 1 : i32
    scf.for %arg15 = %c0_i32 to %c15_i32 step %c1_i32_10  : i32 {
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_FIX>] flag = 1
    }
    return
  }
  func.func @main_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c512_i32 = arith.constant 512 : i32
    %1 = arith.muli %c512_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1024, 512], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<1024x512xf16, strided<[512, 1]>, #hivm.address_space<gm>>
    %c1024_i32 = arith.constant 1024 : i32
    %3 = arith.muli %c1024_i32, %c1_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1024, 1024], strides: [%4, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<1024x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [512, 1024], strides: [%4, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<512x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1024, 1024], strides: [%4, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<1024x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>>
    %5 = hivm.hir.get_block_idx -> i64
    %6 = arith.trunci %5 : i64 to i32
    %7 = hivm.hir.get_sub_block_idx -> i64
    %8 = arith.trunci %7 : i64 to i32
    %alloc = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
    %alloc_3 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
    %c4_i32 = arith.constant 4 : i32
    %9 = arith.divsi %6, %c4_i32 : i32
    %c128_i32 = arith.constant 128 : i32
    %10 = arith.muli %9, %c128_i32 : i32
    %c64_i32 = arith.constant 64 : i32
    %11 = arith.muli %8, %c64_i32 : i32
    %12 = arith.addi %10, %11 : i32
    %13 = arith.remsi %6, %c4_i32 : i32
    %c256_i32 = arith.constant 256 : i32
    %14 = arith.muli %13, %c256_i32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c15_i32 = arith.constant 15 : i32
    %c1_i32_4 = arith.constant 1 : i32
    scf.for %arg15 = %c0_i32 to %c15_i32 step %c1_i32_4  : i32 {
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
    }
    %15 = arith.subi %arg7, %12 : i32
    %16 = arith.minsi %c64_i32, %15 : i32
    %17 = arith.subi %arg8, %14 : i32
    %18 = arith.minsi %c256_i32, %17 : i32
    %c2_i32 = arith.constant 2 : i32
    %19 = arith.remsi %6, %c2_i32 : i32
    %c5_i32 = arith.constant 5 : i32
    %20 = arith.addi %19, %c5_i32 : i32
    %21 = arith.extsi %20 : i32 to i64
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %21
    %22 = arith.index_cast %12 : i32 to index
    %23 = arith.index_cast %16 : i32 to index
    %24 = arith.index_cast %14 : i32 to index
    %25 = arith.index_cast %18 : i32 to index
    %subview = memref.subview %reinterpret_cast_0[%22, %24] [%23, %25] [1, 1] : memref<1024x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_5 = memref.subview %alloc[0, 0] [%23, %25] [1, 1] : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xf16, strided<[256, 1]>, #hivm.address_space<ub>>
    memref.copy %subview, %subview_5 : memref<?x?xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
    hivm.hir.vexp ins(%alloc : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_3 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
    %subview_6 = memref.subview %alloc_3[0, 0] [%23, %25] [1, 1] : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xf16, strided<[256, 1]>, #hivm.address_space<ub>>
    %subview_7 = memref.subview %reinterpret_cast_2[%22, %24] [%23, %25] [1, 1] : memref<1024x1024xf16, strided<[1024, 1]>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_6, %subview_7 : memref<?x?xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xf16, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
    return
  }
}
