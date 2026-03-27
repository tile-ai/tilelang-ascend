// RUN: tilelangir-opt %s -tilelangir-merge-copy-chains -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @cc_cbuf_gm_merges
// CHECK: memref.copy{{.*}}#hivm.address_space<cc>> to memref{{.*}}#hivm.address_space<gm>>
// CHECK: return

module {
  func.func @cc_cbuf_gm_merges(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    hivm.hir.set_ffts_base_addr %arg0
    %gm = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    %cb = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>>
    %a = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cc>>
    memref.copy %a, %cb : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cc>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>>
    memref.copy %cb, %gm : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    return
  }
}

// -----

// CHECK-LABEL: func.func @cc_gm_ub_unchanged
// CHECK: memref.copy{{.*}}#hivm.address_space<cc>> to memref{{.*}}#hivm.address_space<gm>>
// CHECK: memref.copy{{.*}}#hivm.address_space<gm>> to memref{{.*}}#hivm.address_space<ub>>
// CHECK: return

module {
  func.func @cc_gm_ub_unchanged(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    hivm.hir.set_ffts_base_addr %arg0
    %ws = memref_ext.alloc_workspace() : memref<2x2xf32, #hivm.address_space<gm>>
    %a = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cc>>
    %b = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<ub>>
    memref.copy %a, %ws : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cc>> to memref<2x2xf32, #hivm.address_space<gm>>
    memref.copy %ws, %b : memref<2x2xf32, #hivm.address_space<gm>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<ub>>
    return
  }
}

// -----

// CHECK-LABEL: func.func @nested_scf_for_merges
// CHECK: scf.for
// CHECK: memref.copy{{.*}}#hivm.address_space<cc>> to memref{{.*}}#hivm.address_space<gm>>

module {
  func.func @nested_scf_for_merges(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %iv = %c0 to %c1 step %c1 : i32 {
      %gm = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
      %cb = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>>
      %a = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cc>>
      memref.copy %a, %cb : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cc>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>>
      memref.copy %cb, %gm : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @multi_user_cb_all_merge
// CHECK-COUNT-2: memref.copy{{.*}}#hivm.address_space<cc>> to memref{{.*}}#hivm.address_space<gm>>
// CHECK: return

module {
  func.func @multi_user_cb_all_merge(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    hivm.hir.set_ffts_base_addr %arg0
    %gm0 = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    %gm1 = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    %cb = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>>
    %a = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cc>>
    memref.copy %a, %cb : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cc>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>>
    memref.copy %cb, %gm0 : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    memref.copy %cb, %gm1 : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    return
  }
}

// -----

// CHECK-LABEL: func.func @cc_modified_between_no_merge
// CHECK: memref.copy{{.*}}#hivm.address_space<cc>> to memref{{.*}}#hivm.address_space<cbuf>>
// CHECK: memref.copy{{.*}}#hivm.address_space<gm>> to memref{{.*}}#hivm.address_space<cc>>
// CHECK: memref.copy{{.*}}#hivm.address_space<cbuf>> to memref{{.*}}#hivm.address_space<gm>>
// CHECK: return

module {
  func.func @cc_modified_between_no_merge(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    hivm.hir.set_ffts_base_addr %arg0
    %gm = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    %gm2 = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    %cb = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>>
    %a = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cc>>
    memref.copy %a, %cb : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cc>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>>
    memref.copy %gm2, %a : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cc>>
    memref.copy %cb, %gm : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    return
  }
}

// -----

// Multi-user with cc modified between: first cbuf→gm is safe, second is not.

// CHECK-LABEL: func.func @partial_cc_modified
// CHECK: memref.copy{{.*}}#hivm.address_space<cc>> to memref{{.*}}#hivm.address_space<cbuf>>
// CHECK: memref.copy{{.*}}#hivm.address_space<cc>> to memref{{.*}}#hivm.address_space<gm>>
// CHECK: memref.copy{{.*}}#hivm.address_space<gm>> to memref{{.*}}#hivm.address_space<cc>>
// CHECK: memref.copy{{.*}}#hivm.address_space<cbuf>> to memref{{.*}}#hivm.address_space<gm>>
// CHECK: return

module {
  func.func @partial_cc_modified(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    hivm.hir.set_ffts_base_addr %arg0
    %gm0 = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    %gm1 = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    %gm2 = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    %cb = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>>
    %a = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cc>>
    memref.copy %a, %cb : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cc>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>>
    memref.copy %cb, %gm0 : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    memref.copy %gm2, %a : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cc>>
    memref.copy %cb, %gm1 : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    return
  }
}

// -----

// CHECK-LABEL: func.func @two_chains_all_merge
// CHECK-COUNT-3: memref.copy{{.*}}#hivm.address_space<cc>> to memref{{.*}}#hivm.address_space<gm>>
// CHECK: return

module {
  func.func @two_chains_all_merge(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    hivm.hir.set_ffts_base_addr %arg0
    %gm_ok = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    %gm_a = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    %gm_b = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    %cb_ok = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>>
    %cb_multi = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>>
    %a = memref.alloc() : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cc>>
    memref.copy %a, %cb_ok : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cc>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>>
    memref.copy %cb_ok, %gm_ok : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    memref.copy %a, %cb_multi : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cc>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>>
    memref.copy %cb_multi, %gm_a : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    memref.copy %cb_multi, %gm_b : memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<cbuf>> to memref<2x2xf32, strided<[2, 1]>, #hivm.address_space<gm>>
    return
  }
}
