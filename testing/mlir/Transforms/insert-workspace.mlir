// RUN: tilelangir-opt %s -tilelangir-insert-workspace -split-input-file | FileCheck %s

// CHECK-LABEL: @read_interleave
// CHECK: memref_ext.alloc_workspace() : memref<64x64xf32>
// CHECK: hivm.hir.mmadL1
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32>
// CHECK-NEXT: memref.copy %{{.*}}, %{{.*}} : memref<64x64xf32> to memref<64x64xf32, strided<[64, 1]>>
// CHECK: hivm.hir.vmul

// CHECK-NOT: memref.copy %{{.*}}, %{{.*}} : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32>
// CHECK: hivm.hir.mmadL1 ins(%alloc
func.func @read_interleave(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
  %c64 = arith.constant 64 : index
  %true = arith.constant true
  hivm.hir.set_ffts_base_addr %arg0
  %alloc_buf = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  %alloc_src = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
  %alloc_src2 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
  %alloc_scale = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  %alloc_tmp = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  hivm.hir.mmadL1 {b_transpose} ins(%alloc_src, %alloc_src2, %true, %c64, %c64, %c64 : memref<64x128xf16, strided<[128, 1]>>, memref<64x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_buf : memref<64x64xf32, strided<[64, 1]>>)
  hivm.hir.vmul ins(%alloc_buf, %alloc_scale : memref<64x64xf32, strided<[64, 1]>>, memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_tmp : memref<64x64xf32, strided<[64, 1]>>)
  hivm.hir.mmadL1 ins(%alloc_src, %alloc_src2, %true, %c64, %c64, %c64 : memref<64x128xf16, strided<[128, 1]>>, memref<64x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_buf : memref<64x64xf32, strided<[64, 1]>>)
  return
}

// -----

// CHECK-LABEL: @conflicting_writers
// CHECK: memref_ext.alloc_workspace() : memref<64x64xf32>
// CHECK: hivm.hir.mmadL1
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32>
// CHECK-NEXT: memref.copy %{{.*}}, %{{.*}} : memref<64x64xf32> to memref<64x64xf32, strided<[64, 1]>>
// CHECK: hivm.hir.vmul ins(%alloc{{.*}},
// CHECK: memref.copy %alloc{{.*}}, %{{.*}} : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32, strided<[64, 1]>>
func.func @conflicting_writers(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
  %c64 = arith.constant 64 : index
  %true = arith.constant true
  hivm.hir.set_ffts_base_addr %arg0
  %src = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  %dst = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  %a = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
  %b = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
  %scale = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  hivm.hir.mmadL1 {b_transpose} ins(%a, %b, %true, %c64, %c64, %c64 : memref<64x128xf16, strided<[128, 1]>>, memref<64x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%src : memref<64x64xf32, strided<[64, 1]>>)
  hivm.hir.vmul ins(%src, %scale : memref<64x64xf32, strided<[64, 1]>>, memref<64x64xf32, strided<[64, 1]>>) outs(%src : memref<64x64xf32, strided<[64, 1]>>)
  memref.copy %src, %dst : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32, strided<[64, 1]>>
  return
}

// -----

// CHECK-LABEL: @copy_source
// CHECK: memref_ext.alloc_workspace() : memref<64x64xf16>
// CHECK-NOT: memref_ext.alloc_workspace()
// CHECK: hivm.hir.vcast
// CHECK: memref.copy
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<64x64xf16, strided<[64, 1]>> to memref<64x64xf16>
// CHECK-NEXT: memref.copy %{{.*}}, %{{.*}} : memref<64x64xf16> to memref<64x64xf16, strided<[64, 1]>>
// CHECK: hivm.hir.mmadL1
func.func @copy_source(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
  %c64 = arith.constant 64 : index
  %true = arith.constant true
  hivm.hir.set_ffts_base_addr %arg0
  %buf = memref.alloc() : memref<32x64xf16, strided<[64, 1]>>
  %src = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
  %target = memref.alloc() : memref<64x64xf16, strided<[64, 1]>>
  %other = memref.alloc() : memref<64x64xf16, strided<[64, 1]>>
  %out = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  hivm.hir.vcast ins(%src : memref<32x64xf32, strided<[64, 1]>>) outs(%buf : memref<32x64xf16, strided<[64, 1]>>)
  %subview = memref.subview %target[0, 0] [32, 64] [1, 1] : memref<64x64xf16, strided<[64, 1]>> to memref<32x64xf16, strided<[64, 1]>>
  memref.copy %buf, %subview : memref<32x64xf16, strided<[64, 1]>> to memref<32x64xf16, strided<[64, 1]>>
  hivm.hir.mmadL1 {b_transpose} ins(%target, %other, %true, %c64, %c64, %c64 : memref<64x64xf16, strided<[64, 1]>>, memref<64x64xf16, strided<[64, 1]>>, i1, index, index, index) outs(%out : memref<64x64xf32, strided<[64, 1]>>)
  return
}

// -----

// CHECK-LABEL: @minicv
// CHECK: %[[WS:.*]] = memref_ext.alloc_workspace() : memref<64x32xf16>
// CHECK: memref.copy %{{.*}}, %[[WS]] : memref<64x32xf16, strided<[32, 1]>> to memref<64x32xf16>
// CHECK-NEXT: memref.copy %[[WS]], %{{.*}} : memref<64x32xf16> to memref<64x32xf16, strided<[32, 1]>>
func.func @minicv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c768 = arith.constant 768 : index
  %c1 = arith.constant 1 : index
  %c32_i32 = arith.constant 32 : i32
  %c64_i32 = arith.constant 64 : i32
  %c2_i32 = arith.constant 2 : i32
  %c24_i32 = arith.constant 24 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  hivm.hir.set_ffts_base_addr %arg0
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [128, 768], strides: [%c768, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x768xf16, strided<[768, 1]>, #hivm.address_space<gm>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [768, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<768x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [128, 128], strides: [%c128, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<gm>>
  %0 = hivm.hir.get_block_idx -> i64
  %1 = arith.trunci %0 : i64 to i32
  %2 = hivm.hir.get_sub_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %alloc = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  scf.for %arg13 = %c0_i32 to %c24_i32 step %c1_i32  : i32 {
    %alloc_2 = memref.alloc() : memref<32x32xf16, strided<[32, 1]>>
    %alloc_3 = memref.alloc() : memref<32x32xf16, strided<[32, 1]>>
    %alloc_4 = memref.alloc() : memref<64x32xf16, strided<[32, 1]>>
    %alloc_5 = memref.alloc() : memref<32x64xf16, strided<[64, 1]>>
    %10 = arith.divsi %1, %c2_i32 : i32
    %11 = arith.muli %10, %c64_i32 : i32
    %12 = arith.muli %3, %c32_i32 : i32
    %13 = arith.addi %11, %12 : i32
    %14 = arith.index_cast %13 : i32 to index
    %15 = arith.muli %arg13, %c32_i32 : i32
    %16 = arith.index_cast %15 : i32 to index
    %subview_6 = memref.subview %reinterpret_cast[%14, %16] [32, 32] [1, 1] : memref<128x768xf16, strided<[768, 1]>, #hivm.address_space<gm>> to memref<32x32xf16, strided<[768, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_6, %alloc_2 : memref<32x32xf16, strided<[768, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x32xf16, strided<[32, 1]>>
    %17 = arith.remsi %1, %c2_i32 : i32
    %18 = arith.muli %17, %c64_i32 : i32
    %19 = arith.index_cast %18 : i32 to index
    %subview_7 = memref.subview %reinterpret_cast_0[%16, %19] [32, 64] [1, 1] : memref<768x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<32x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_7, %alloc_5 : memref<32x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x64xf16, strided<[64, 1]>>
    hivm.hir.vexp ins(%alloc_2 : memref<32x32xf16, strided<[32, 1]>>) outs(%alloc_3 : memref<32x32xf16, strided<[32, 1]>>)
    %20 = arith.index_cast %12 : i32 to index
    %subview_8 = memref.subview %alloc_4[%20, 0] [32, 32] [1, 1] : memref<64x32xf16, strided<[32, 1]>> to memref<32x32xf16, strided<[32, 1], offset: ?>>
    memref.copy %alloc_3, %subview_8 : memref<32x32xf16, strided<[32, 1]>> to memref<32x32xf16, strided<[32, 1], offset: ?>>
    %21 = arith.cmpi eq, %arg13, %c0_i32 : i32
    hivm.hir.mmadL1 ins(%alloc_4, %alloc_5, %21, %c64, %c32, %c64 : memref<64x32xf16, strided<[32, 1]>>, memref<32x64xf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc : memref<64x64xf32, strided<[64, 1]>>)
  }
  %4 = arith.divsi %1, %c2_i32 : i32
  %5 = arith.muli %4, %c64_i32 : i32
  %6 = arith.index_cast %5 : i32 to index
  %7 = arith.remsi %1, %c2_i32 : i32
  %8 = arith.muli %7, %c64_i32 : i32
  %9 = arith.index_cast %8 : i32 to index
  %subview = memref.subview %reinterpret_cast_1[%6, %9] [64, 64] [1, 1] : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
  memref.copy %alloc, %subview : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
  return
}

// -----

// CHECK-LABEL: @flash_attention
//
// CHECK-DAG: memref_ext.alloc_workspace() : memref<64x64xf32>
// CHECK-DAG: memref_ext.alloc_workspace() : memref<64x64xf16>
// CHECK-DAG: memref_ext.alloc_workspace() : memref<64x128xf32>
//
// Boundary 1: alloc_13 (CUBE mmadL1 -> VECTOR vmul)
// CHECK: hivm.hir.mmadL1 {b_transpose}
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32>
// CHECK-NEXT: memref.copy %{{.*}}, %{{.*}} : memref<64x64xf32> to memref<64x64xf32, strided<[64, 1]>>
// CHECK: hivm.hir.vmul
//
// Boundary 2: alloc_14 (VECTOR vcast -> CUBE mmadL1)
// CHECK: hivm.hir.vcast
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<64x64xf16, strided<[64, 1]>> to memref<64x64xf16>
// CHECK-NEXT: memref.copy %{{.*}}, %{{.*}} : memref<64x64xf16> to memref<64x64xf16, strided<[64, 1]>>
//
// Boundary 3: alloc_18 (CUBE mmadL1 -> VECTOR vadd)
// CHECK: hivm.hir.mmadL1 ins
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<64x128xf32, strided<[128, 1]>> to memref<64x128xf32>
// CHECK-NEXT: memref.copy %{{.*}}, %{{.*}} : memref<64x128xf32> to memref<64x128xf32, strided<[128, 1]>>
// CHECK: hivm.hir.vadd
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
  %alloc = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
  %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
  %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
  %alloc_7 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
  %alloc_8 = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  %alloc_9 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
  %2 = arith.muli %1, %c64_i32 : i32
  %3 = arith.index_cast %2 : i32 to index
  %subview = memref.subview %reinterpret_cast[%3, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
  memref.copy %subview, %alloc : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1]>>
  hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>)
  hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_8 : memref<64x64xf32, strided<[64, 1]>>)
  scf.for %arg13 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
    %alloc_11 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
    %alloc_12 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
    %alloc_13 = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
    %alloc_14 = memref.alloc() : memref<64x64xf16, strided<[64, 1]>>
    %alloc_15 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
    %alloc_16 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
    %alloc_17 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
    %alloc_18 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
    %alloc_19 = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
    %alloc_20 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
    %alloc_21 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
    %4 = arith.muli %arg13, %c64_i32 : i32
    %5 = arith.index_cast %4 : i32 to index
    %subview_22 = memref.subview %reinterpret_cast_3[%5, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_22, %alloc_11 : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1]>>
    hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_11, %true, %c64, %c64, %c128 : memref<64x128xf16, strided<[128, 1]>>, memref<64x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_13 : memref<64x64xf32, strided<[64, 1]>>)
    hivm.hir.vmul ins(%alloc_13, %alloc_8 : memref<64x64xf32, strided<[64, 1]>>, memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_13 : memref<64x64xf32, strided<[64, 1]>>)
    hivm.hir.vreduce <max> ins(%alloc_13 : memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_16 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
    hivm.hir.vmax ins(%alloc_5, %alloc_16 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_21 : memref<64x1xf32, strided<[1, 1]>>)
    hivm.hir.vsub ins(%alloc_5, %alloc_21 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_20 : memref<64x1xf32, strided<[1, 1]>>)
    hivm.hir.vexp ins(%alloc_20 : memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_15 : memref<64x1xf32, strided<[1, 1]>>)
    hivm.hir.vsub ins(%alloc_13, %alloc_21 : memref<64x64xf32, strided<[64, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_19 : memref<64x64xf32, strided<[64, 1]>>) broadcast = [1]
    hivm.hir.vexp ins(%alloc_19 : memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_13 : memref<64x64xf32, strided<[64, 1]>>)
    hivm.hir.vreduce <sum> ins(%alloc_13 : memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
    hivm.hir.vmul ins(%alloc_6, %alloc_15 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
    hivm.hir.vadd ins(%alloc_6, %alloc_17 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
    hivm.hir.vmul ins(%alloc_7, %alloc_15 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
    hivm.hir.vcast ins(%alloc_13 : memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_14 : memref<64x64xf16, strided<[64, 1]>>)
    hivm.hir.vbrc ins(%cst : f32) outs(%alloc_20 : memref<64x1xf32, strided<[1, 1]>>)
    hivm.hir.vadd ins(%alloc_20, %alloc_21 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
    %subview_23 = memref.subview %reinterpret_cast_2[%5, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_23, %alloc_12 : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1]>>
    hivm.hir.mmadL1 ins(%alloc_14, %alloc_12, %true, %c64, %c64, %c128 : memref<64x64xf16, strided<[64, 1]>>, memref<64x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_18 : memref<64x128xf32, strided<[128, 1]>>)
    hivm.hir.vadd ins(%alloc_18, %alloc_7 : memref<64x128xf32, strided<[128, 1]>>, memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>)
  }
  hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
  hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_9 : memref<64x128xf16, strided<[128, 1]>>)
  %subview_10 = memref.subview %reinterpret_cast_4[%3, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
  memref.copy %alloc_9, %subview_10 : memref<64x128xf16, strided<[128, 1]>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
  return
}
