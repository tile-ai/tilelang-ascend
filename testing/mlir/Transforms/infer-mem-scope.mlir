// RUN: tilelangir-opt %s -tilelangir-infer-mem-scope -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @read_interleave
// CHECK-SAME: %arg1: memref<?xi8, #hivm.address_space<gm>>
// CHECK-SAME: %arg2: memref<?xi8, #hivm.address_space<gm>>
// CHECK-SAME: %arg3: memref<?xf16, #hivm.address_space<gm>>

// CHECK:      memref_ext.alloc_workspace() : memref<64x64xf32, #hivm.address_space<gm>>
// CHECK:      memref.alloc() : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>
// CHECK:      memref.alloc() : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
// CHECK:      memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
// CHECK:      memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
// CHECK:      memref.alloc() : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
// CHECK:      memref.alloc() : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>

// mmadL1: mA,mB→cbuf, mC→cc
// CHECK:      hivm.hir.mmadL1
// CHECK-SAME:   #hivm.address_space<cbuf>>
// CHECK-SAME:   #hivm.address_space<cbuf>>
// CHECK-SAME:   #hivm.address_space<cc>>

// copy: cc→gm (fixpipe semantics)
// CHECK:      memref.copy {{.*}}#hivm.address_space<cc>> to {{.*}}#hivm.address_space<gm>>
// copy: gm→ub (MTE2 load)
// CHECK:      memref.copy {{.*}}#hivm.address_space<gm>> to {{.*}}#hivm.address_space<ub>>

// vmul: all operands ub
// CHECK:      hivm.hir.vmul
// CHECK-SAME:   #hivm.address_space<ub>>
// CHECK-SAME:   #hivm.address_space<ub>>
// CHECK-SAME:   #hivm.address_space<ub>>

func.func @read_interleave(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
  %c64 = arith.constant 64 : index
  %true = arith.constant true
  hivm.hir.set_ffts_base_addr %arg0
  %0 = memref_ext.alloc_workspace() : memref<64x64xf32>
  %alloc = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  %alloc_0 = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  %alloc_1 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
  %alloc_2 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
  %alloc_3 = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  %alloc_4 = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  hivm.hir.mmadL1 {b_transpose} ins(%alloc_1, %alloc_2, %true, %c64, %c64, %c64 : memref<64x128xf16, strided<[128, 1]>>, memref<64x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc : memref<64x64xf32, strided<[64, 1]>>)
  memref.copy %alloc, %0 : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32>
  memref.copy %0, %alloc_0 : memref<64x64xf32> to memref<64x64xf32, strided<[64, 1]>>
  hivm.hir.vmul ins(%alloc_0, %alloc_3 : memref<64x64xf32, strided<[64, 1]>>, memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_4 : memref<64x64xf32, strided<[64, 1]>>)
  hivm.hir.mmadL1 ins(%alloc_1, %alloc_2, %true, %c64, %c64, %c64 : memref<64x128xf16, strided<[128, 1]>>, memref<64x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc : memref<64x64xf32, strided<[64, 1]>>)
  return
}

// -----

// CHECK-LABEL: func.func @conflicting_writers
// CHECK-SAME: %arg1: memref<?xi8, #hivm.address_space<gm>>
// CHECK-SAME: %arg2: memref<?xi8, #hivm.address_space<gm>>

// CHECK:      memref_ext.alloc_workspace() : memref<64x64xf32, #hivm.address_space<gm>>
// CHECK:      memref.alloc() : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>
// CHECK:      memref.alloc() : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
// Phase 5: remaining alloc in AIC function → cbuf (L1 default)
// CHECK:      memref.alloc() : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
// CHECK:      memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
// CHECK:      memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
// CHECK:      memref.alloc() : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>

// Copy from ub to cbuf (cross-address-space)
// CHECK:      memref.copy {{.*}}#hivm.address_space<ub>> to {{.*}}#hivm.address_space<cbuf>>

func.func @conflicting_writers(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
  %c64 = arith.constant 64 : index
  %true = arith.constant true
  hivm.hir.set_ffts_base_addr %arg0
  %0 = memref_ext.alloc_workspace() : memref<64x64xf32>
  %alloc = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  %alloc_0 = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  %alloc_1 = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  %alloc_2 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
  %alloc_3 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
  %alloc_4 = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  hivm.hir.mmadL1 {b_transpose} ins(%alloc_2, %alloc_3, %true, %c64, %c64, %c64 : memref<64x128xf16, strided<[128, 1]>>, memref<64x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc : memref<64x64xf32, strided<[64, 1]>>)
  memref.copy %alloc, %0 : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32>
  memref.copy %0, %alloc_0 : memref<64x64xf32> to memref<64x64xf32, strided<[64, 1]>>
  hivm.hir.vmul ins(%alloc_0, %alloc_4 : memref<64x64xf32, strided<[64, 1]>>, memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_0 : memref<64x64xf32, strided<[64, 1]>>)
  memref.copy %alloc_0, %alloc_1 : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32, strided<[64, 1]>>
  return
}

// -----

// CHECK-LABEL: func.func @minicv
// CHECK-SAME: %arg1: memref<?xi8, #hivm.address_space<gm>>
// CHECK-SAME: %arg2: memref<?xi8, #hivm.address_space<gm>>

// CHECK:      memref_ext.alloc_workspace() : memref<64x32xf16, #hivm.address_space<gm>>
// CHECK:      memref.alloc() : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>

// VECTOR scope: all allocs→ub
// CHECK:      scope.scope
// CHECK:        memref.alloc() : memref<32x32xf16, strided<[32, 1]>, #hivm.address_space<ub>>
// CHECK:        memref.alloc() : memref<32x32xf16, strided<[32, 1]>, #hivm.address_space<ub>>
// CHECK:        memref.alloc() : memref<64x32xf16, strided<[32, 1]>, #hivm.address_space<ub>>
// CHECK:        memref.copy {{.*}}#hivm.address_space<gm>> to {{.*}}#hivm.address_space<ub>>
// CHECK:        hivm.hir.vexp
// CHECK-SAME:     #hivm.address_space<ub>>
// CHECK-SAME:     #hivm.address_space<ub>>
// CHECK:        memref.subview {{.*}}#hivm.address_space<ub>>
// CHECK:        memref.copy {{.*}}#hivm.address_space<ub>> to {{.*}}#hivm.address_space<ub>>
// CHECK:        memref.copy {{.*}}#hivm.address_space<ub>> to {{.*}}#hivm.address_space<gm>>
// CHECK:      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

// CUBE scope: allocs→cbuf (from mmadL1), output→cc
// CHECK:      scope.scope
// CHECK:        memref.alloc() : memref<64x32xf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
// CHECK:        memref.alloc() : memref<32x64xf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
// CHECK:        memref.copy {{.*}}#hivm.address_space<gm>> to {{.*}}#hivm.address_space<cbuf>>
// CHECK:        memref.copy {{.*}}#hivm.address_space<gm>> to {{.*}}#hivm.address_space<cbuf>>
// CHECK:        hivm.hir.mmadL1
// CHECK-SAME:     #hivm.address_space<cbuf>>
// CHECK-SAME:     #hivm.address_space<cbuf>>
// CHECK-SAME:     #hivm.address_space<cc>>
// CHECK:      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}

// Tail: fixpipe copy cc→gm
// CHECK:      memref.copy {{.*}}#hivm.address_space<cc>> to {{.*}}#hivm.address_space<gm>>

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
  %4 = memref_ext.alloc_workspace() : memref<64x32xf16>
  %alloc = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
  scf.for %arg13 = %c0_i32 to %c24_i32 step %c1_i32  : i32 {
    %11 = arith.divsi %1, %c2_i32 : i32
    %12 = arith.muli %11, %c64_i32 : i32
    %13 = arith.muli %3, %c32_i32 : i32
    %14 = arith.addi %12, %13 : i32
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %arg13, %c32_i32 : i32
    %17 = arith.index_cast %16 : i32 to index
    %18 = arith.remsi %1, %c2_i32 : i32
    %19 = arith.muli %18, %c64_i32 : i32
    %20 = arith.index_cast %19 : i32 to index
    %21 = arith.index_cast %13 : i32 to index
    %22 = arith.cmpi eq, %arg13, %c0_i32 : i32
    scope.scope : () -> () {
      %alloc_2 = memref.alloc() : memref<32x32xf16, strided<[32, 1]>>
      %alloc_3 = memref.alloc() : memref<32x32xf16, strided<[32, 1]>>
      %alloc_4 = memref.alloc() : memref<64x32xf16, strided<[32, 1]>>
      %subview_5 = memref.subview %reinterpret_cast[%15, %17] [32, 32] [1, 1] : memref<128x768xf16, strided<[768, 1]>, #hivm.address_space<gm>> to memref<32x32xf16, strided<[768, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_5, %alloc_2 : memref<32x32xf16, strided<[768, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x32xf16, strided<[32, 1]>>
      hivm.hir.vexp ins(%alloc_2 : memref<32x32xf16, strided<[32, 1]>>) outs(%alloc_3 : memref<32x32xf16, strided<[32, 1]>>)
      %subview_6 = memref.subview %alloc_4[%21, 0] [32, 32] [1, 1] : memref<64x32xf16, strided<[32, 1]>> to memref<32x32xf16, strided<[32, 1], offset: ?>>
      memref.copy %alloc_3, %subview_6 : memref<32x32xf16, strided<[32, 1]>> to memref<32x32xf16, strided<[32, 1], offset: ?>>
      memref.copy %alloc_4, %4 : memref<64x32xf16, strided<[32, 1]>> to memref<64x32xf16>
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %alloc_2 = memref.alloc() : memref<64x32xf16, strided<[32, 1]>>
      %alloc_3 = memref.alloc() : memref<32x64xf16, strided<[64, 1]>>
      %subview_4 = memref.subview %reinterpret_cast_0[%17, %20] [32, 64] [1, 1] : memref<768x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<32x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_4, %alloc_3 : memref<32x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x64xf16, strided<[64, 1]>>
      memref.copy %4, %alloc_2 : memref<64x32xf16> to memref<64x32xf16, strided<[32, 1]>>
      hivm.hir.mmadL1 ins(%alloc_2, %alloc_3, %22, %c64, %c32, %c64 : memref<64x32xf16, strided<[32, 1]>>, memref<32x64xf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc : memref<64x64xf32, strided<[64, 1]>>)
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  }
  %5 = arith.divsi %1, %c2_i32 : i32
  %6 = arith.muli %5, %c64_i32 : i32
  %7 = arith.index_cast %6 : i32 to index
  %8 = arith.remsi %1, %c2_i32 : i32
  %9 = arith.muli %8, %c64_i32 : i32
  %10 = arith.index_cast %9 : i32 to index
  %subview = memref.subview %reinterpret_cast_1[%7, %10] [64, 64] [1, 1] : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
  memref.copy %alloc, %subview : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
  return
}

// -----

// CHECK-LABEL: func.func @flash_attention
// CHECK-SAME: %arg1: memref<?xi8, #hivm.address_space<gm>>
// CHECK-SAME: %arg2: memref<?xi8, #hivm.address_space<gm>>

// CHECK-DAG:  memref_ext.alloc_workspace() : memref<64x64xf32, #hivm.address_space<gm>>
// CHECK-DAG:  memref_ext.alloc_workspace() : memref<64x64xf16, #hivm.address_space<gm>>
// CHECK-DAG:  memref_ext.alloc_workspace() : memref<64x128xf32, #hivm.address_space<gm>>

// Q (mmadL1 mA) → cbuf; VECTOR allocs → ub
// CHECK:      memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
// CHECK:      memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
// CHECK:      memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
// CHECK:      memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
// CHECK:      memref.alloc() : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
// CHECK:      memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>

// Load Q: gm→cbuf
// CHECK:      memref.copy {{.*}}#hivm.address_space<gm>> to {{.*}}#hivm.address_space<cbuf>>

// vbrc outputs → ub
// CHECK:      hivm.hir.vbrc {{.*}} outs({{.*}}#hivm.address_space<ub>>)
// CHECK:      hivm.hir.vbrc {{.*}} outs({{.*}}#hivm.address_space<ub>>)
// CHECK:      hivm.hir.vbrc {{.*}} outs({{.*}}#hivm.address_space<ub>>)
// CHECK:      hivm.hir.vbrc {{.*}} outs({{.*}}#hivm.address_space<ub>>)

// CUBE scope 1: K→cbuf, S→cc, copy S→gm
// CHECK:      scope.scope
// CHECK:        memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
// CHECK:        memref.alloc() : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>
// CHECK:        memref.copy {{.*}}#hivm.address_space<gm>> to {{.*}}#hivm.address_space<cbuf>>
// CHECK:        hivm.hir.mmadL1
// CHECK-SAME:     #hivm.address_space<cbuf>>
// CHECK-SAME:     #hivm.address_space<cbuf>>
// CHECK-SAME:     #hivm.address_space<cc>>
// CHECK:        memref.copy {{.*}}#hivm.address_space<cc>> to {{.*}}#hivm.address_space<gm>>
// CHECK:      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}

// VECTOR scope: all allocs→ub, softmax in ub
// CHECK:      scope.scope
// CHECK:        memref.alloc() : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
// CHECK:        memref.alloc() : memref<64x64xf16, strided<[64, 1]>, #hivm.address_space<ub>>
// CHECK:        memref.copy {{.*}}#hivm.address_space<gm>> to {{.*}}#hivm.address_space<ub>>
// CHECK:        hivm.hir.vmul
// CHECK-SAME:     #hivm.address_space<ub>>
// CHECK:        hivm.hir.vcast
// CHECK-SAME:     #hivm.address_space<ub>>
// CHECK-SAME:     #hivm.address_space<ub>>
// CHECK:        memref.copy {{.*}}#hivm.address_space<ub>> to {{.*}}#hivm.address_space<gm>>
// CHECK:      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

// CUBE scope 2: P→cbuf, V→cbuf, O'→cc
// CHECK:      scope.scope
// CHECK:        memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
// CHECK:        memref.alloc() : memref<64x64xf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
// CHECK:        memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
// CHECK:        hivm.hir.mmadL1
// CHECK-SAME:     #hivm.address_space<cbuf>>
// CHECK-SAME:     #hivm.address_space<cbuf>>
// CHECK-SAME:     #hivm.address_space<cc>>
// CHECK:        memref.copy {{.*}}#hivm.address_space<cc>> to {{.*}}#hivm.address_space<gm>>
// CHECK:      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}

// VECTOR scope 2: accumulate O'
// CHECK:      scope.scope
// CHECK:        memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
// CHECK:        memref.copy {{.*}}#hivm.address_space<gm>> to {{.*}}#hivm.address_space<ub>>
// CHECK:        hivm.hir.vadd
// CHECK-SAME:     #hivm.address_space<ub>>
// CHECK-SAME:     #hivm.address_space<ub>>
// CHECK-SAME:     #hivm.address_space<ub>>
// CHECK:      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

// Final normalize and write back: ub→gm
// CHECK:      hivm.hir.vdiv
// CHECK-SAME:   #hivm.address_space<ub>>
// CHECK:      hivm.hir.vcast
// CHECK-SAME:   #hivm.address_space<ub>>
// CHECK-SAME:   #hivm.address_space<ub>>
// CHECK:      memref.copy {{.*}}#hivm.address_space<ub>> to {{.*}}#hivm.address_space<gm>>

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
  %2 = memref_ext.alloc_workspace() : memref<64x64xf32>
  %3 = memref_ext.alloc_workspace() : memref<64x64xf16>
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
    %7 = arith.muli %arg13, %c64_i32 : i32
    %8 = arith.index_cast %7 : i32 to index
    scope.scope : () -> () {
      %alloc_11 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
      %alloc_12 = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
      %subview_13 = memref.subview %reinterpret_cast_3[%8, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_13, %alloc_11 : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_11, %true, %c64, %c64, %c128 : memref<64x128xf16, strided<[128, 1]>>, memref<64x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_12 : memref<64x64xf32, strided<[64, 1]>>)
      memref.copy %alloc_12, %2 : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32>
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      %alloc_11 = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
      %alloc_12 = memref.alloc() : memref<64x64xf16, strided<[64, 1]>>
      %alloc_13 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_14 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_15 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_16 = memref.alloc() : memref<64x64xf32, strided<[64, 1]>>
      %alloc_17 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_18 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      memref.copy %2, %alloc_11 : memref<64x64xf32> to memref<64x64xf32, strided<[64, 1]>>
      hivm.hir.vmul ins(%alloc_11, %alloc_8 : memref<64x64xf32, strided<[64, 1]>>, memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_11 : memref<64x64xf32, strided<[64, 1]>>)
      hivm.hir.vreduce <max> ins(%alloc_11 : memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_14 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
      hivm.hir.vmax ins(%alloc_5, %alloc_14 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_18 : memref<64x1xf32, strided<[1, 1]>>)
      hivm.hir.vsub ins(%alloc_5, %alloc_18 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>>)
      hivm.hir.vexp ins(%alloc_17 : memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_13 : memref<64x1xf32, strided<[1, 1]>>)
      hivm.hir.vsub ins(%alloc_11, %alloc_18 : memref<64x64xf32, strided<[64, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_16 : memref<64x64xf32, strided<[64, 1]>>) broadcast = [1]
      hivm.hir.vexp ins(%alloc_16 : memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_11 : memref<64x64xf32, strided<[64, 1]>>)
      hivm.hir.vreduce <sum> ins(%alloc_11 : memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_15 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
      hivm.hir.vmul ins(%alloc_6, %alloc_13 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
      hivm.hir.vadd ins(%alloc_6, %alloc_15 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
      hivm.hir.vmul ins(%alloc_7, %alloc_13 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<64x64xf32, strided<[64, 1]>>) outs(%alloc_12 : memref<64x64xf16, strided<[64, 1]>>)
      memref.copy %alloc_12, %3 : memref<64x64xf16, strided<[64, 1]>> to memref<64x64xf16>
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>>)
      hivm.hir.vadd ins(%alloc_17, %alloc_18 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %alloc_11 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
      %alloc_12 = memref.alloc() : memref<64x64xf16, strided<[64, 1]>>
      %alloc_13 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
      memref.copy %3, %alloc_12 : memref<64x64xf16> to memref<64x64xf16, strided<[64, 1]>>
      %subview_14 = memref.subview %reinterpret_cast_2[%8, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_14, %alloc_11 : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1]>>
      hivm.hir.mmadL1 ins(%alloc_12, %alloc_11, %true, %c64, %c64, %c128 : memref<64x64xf16, strided<[64, 1]>>, memref<64x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_13 : memref<64x128xf32, strided<[128, 1]>>)
      memref.copy %alloc_13, %4 : memref<64x128xf32, strided<[128, 1]>> to memref<64x128xf32>
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      %alloc_11 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
      memref.copy %4, %alloc_11 : memref<64x128xf32> to memref<64x128xf32, strided<[128, 1]>>
      hivm.hir.vadd ins(%alloc_11, %alloc_7 : memref<64x128xf32, strided<[128, 1]>>, memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>)
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  }
  hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
  hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_9 : memref<64x128xf16, strided<[128, 1]>>)
  %subview_10 = memref.subview %reinterpret_cast_4[%6, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
  memref.copy %alloc_9, %subview_10 : memref<64x128xf16, strided<[128, 1]>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
  return
}
