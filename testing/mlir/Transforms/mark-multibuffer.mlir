// RUN: tilelangir-opt %s -tilelangir-mark-multibuffer | FileCheck %s

// CHECK-LABEL: func.func @flash_attention
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c8_i32 = arith.constant 8 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [512, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<512x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [512, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<512x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [512, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<512x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [512, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<512x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    // CHECK: %[[WS0:.*]] = memref_ext.alloc_workspace() : memref<64x64xf32>
    // CHECK-NEXT: annotation.mark %[[WS0]] {hivm.multi_buffer = 2 : i32}
    %2 = memref_ext.alloc_workspace() : memref<64x64xf32>
    // CHECK: %[[WS1:.*]] = memref_ext.alloc_workspace() : memref<64x64xf16>
    // CHECK-NEXT: annotation.mark %[[WS1]] {hivm.multi_buffer = 2 : i32}
    %3 = memref_ext.alloc_workspace() : memref<64x64xf16>
    // CHECK: %[[WS2:.*]] = memref_ext.alloc_workspace() : memref<64x128xf32>
    // CHECK-NEXT: annotation.mark %[[WS2]] {hivm.multi_buffer = 2 : i32}
    %4 = memref_ext.alloc_workspace() : memref<64x128xf32>
    %alloc = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
    %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
    %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
    %alloc_7 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
    %alloc_8 = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
    %alloc_9 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
    %5 = arith.muli %1, %c64_i32 : i32
    %6 = arith.index_cast %5 : i32 to index
    %subview = memref.subview %reinterpret_cast[%6, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1]>>
    hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>)
    hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
    hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
    hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_8 : memref<64x64xf32, strided<[64, 1]>>)
    scf.for %arg13 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      %alloc_11 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
      %alloc_12 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
      %alloc_13 = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
      %alloc_14 = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
      %alloc_15 = memref.alloc() : memref<64x64xf16, strided<[64, 1]>>
      %alloc_16 = memref.alloc() : memref<64x64xf16, strided<[64, 1]>>
      %alloc_17 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_18 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_19 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_20 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
      %alloc_21 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
      %alloc_22 = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
      %alloc_23 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_24 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %7 = arith.muli %arg13, %c64_i32 : i32
      %8 = arith.index_cast %7 : i32 to index
      %subview_25 = memref.subview %reinterpret_cast_3[%8, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_25, %alloc_11 : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_11, %true, %c64, %c64, %c128 : memref<64x128xf16, strided<[128, 1]>>, memref<64x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_13 : memref<64x64xf32, strided<[64, 1]>>)
      memref.copy %alloc_13, %2 : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32>
      memref.copy %2, %alloc_14 : memref<64x64xf32> to memref<64x64xf32, strided<[64, 1]>>
      hivm.hir.vmul ins(%alloc_14, %alloc_8 : memref<64x64xf32, strided<[64, 1]>>, memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_14 : memref<64x64xf32, strided<[64, 1]>>)
      hivm.hir.vreduce <max> ins(%alloc_14 : memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_18 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
      hivm.hir.vmax ins(%alloc_5, %alloc_18 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_24 : memref<64x1xf32, strided<[1, 1]>>)
      hivm.hir.vsub ins(%alloc_5, %alloc_24 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_23 : memref<64x1xf32, strided<[1, 1]>>)
      hivm.hir.vexp ins(%alloc_23 : memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>>)
      hivm.hir.vsub ins(%alloc_14, %alloc_24 : memref<64x64xf32, strided<[64, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_22 : memref<64x64xf32, strided<[64, 1]>>) broadcast = [1]
      hivm.hir.vexp ins(%alloc_22 : memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_14 : memref<64x64xf32, strided<[64, 1]>>)
      hivm.hir.vreduce <sum> ins(%alloc_14 : memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_19 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
      hivm.hir.vmul ins(%alloc_6, %alloc_17 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
      hivm.hir.vadd ins(%alloc_6, %alloc_19 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
      hivm.hir.vmul ins(%alloc_7, %alloc_17 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_14 : memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_15 : memref<64x64xf16, strided<[64, 1]>>)
      memref.copy %alloc_15, %3 : memref<64x64xf16, strided<[64, 1]>> to memref<64x64xf16>
      memref.copy %3, %alloc_16 : memref<64x64xf16> to memref<64x64xf16, strided<[64, 1]>>
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_23 : memref<64x1xf32, strided<[1, 1]>>)
      hivm.hir.vadd ins(%alloc_23, %alloc_24 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
      %subview_26 = memref.subview %reinterpret_cast_2[%8, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_26, %alloc_12 : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1]>>
      hivm.hir.mmadL1 ins(%alloc_16, %alloc_12, %true, %c64, %c64, %c128 : memref<64x64xf16, strided<[64, 1]>>, memref<64x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_20 : memref<64x128xf32, strided<[128, 1]>>)
      memref.copy %alloc_20, %4 : memref<64x128xf32, strided<[128, 1]>> to memref<64x128xf32>
      memref.copy %4, %alloc_21 : memref<64x128xf32> to memref<64x128xf32, strided<[128, 1]>>
      hivm.hir.vadd ins(%alloc_21, %alloc_7 : memref<64x128xf32, strided<[128, 1]>>, memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>)
    } {tilelangir.num_stages = 2 : i32}
    hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
    hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_9 : memref<64x128xf16, strided<[128, 1]>>)
    %subview_10 = memref.subview %reinterpret_cast_4[%6, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %alloc_9, %subview_10 : memref<64x128xf16, strided<[128, 1]>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    return
  }
}
