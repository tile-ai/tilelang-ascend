// RUN: tilelangir-opt %s -tilelangir-plan-workspace-memory -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @three_workspaces
// CHECK: %[[OFF0:.*]] = arith.constant 0 : index
// CHECK-NEXT: memref_ext.alloc_workspace() from %arg2 offset = [%[[OFF0]]] {{.*}} to memref<3x8x64x64xf32
// CHECK: %[[OFF1:.*]] = arith.constant 393216 : index
// CHECK-NEXT: memref_ext.alloc_workspace() from %arg2 offset = [%[[OFF1]]] {{.*}} to memref<3x8x64x64xf16
// CHECK: %[[OFF2:.*]] = arith.constant 589824 : index
// CHECK-NEXT: memref_ext.alloc_workspace() from %arg2 offset = [%[[OFF2]]] {{.*}} to memref<3x8x64x128xf32
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @three_workspaces(
      %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
      %arg1: memref<?xi8, #hivm.address_space<gm>>,
      %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>},
      %arg3: memref<?xf16, #hivm.address_space<gm>>
  ) attributes {
      SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
      hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"
  } {
    // 3*8*64*64*4 = 393216 bytes
    %0 = memref_ext.alloc_workspace() from %arg2
        : from memref<?xi8, #hivm.address_space<gm>>
           to memref<3x8x64x64xf32, #hivm.address_space<gm>>
    // 3*8*64*64*2 = 196608 bytes
    %1 = memref_ext.alloc_workspace() from %arg2
        : from memref<?xi8, #hivm.address_space<gm>>
           to memref<3x8x64x64xf16, #hivm.address_space<gm>>
    // 3*8*64*128*4 = 786432 bytes
    %2 = memref_ext.alloc_workspace() from %arg2
        : from memref<?xi8, #hivm.address_space<gm>>
           to memref<3x8x64x128xf32, #hivm.address_space<gm>>
    return
  }
}

// -----

// CHECK-LABEL: func.func @single_workspace
// CHECK: %[[OFF:.*]] = arith.constant 0 : index
// CHECK: memref_ext.alloc_workspace() from %arg2 offset = [%[[OFF]]] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x128xf32, #hivm.address_space<gm>>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @single_workspace(
      %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
      %arg1: memref<?xi8, #hivm.address_space<gm>>,
      %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}
  ) attributes {
      SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
      hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"
  } {
    // 64*128*4 = 32768 bytes
    %0 = memref_ext.alloc_workspace() from %arg2
        : from memref<?xi8, #hivm.address_space<gm>>
           to memref<64x128xf32, #hivm.address_space<gm>>
    return
  }
}

// -----

// CHECK-LABEL: func.func @already_has_offset
// CHECK: %[[EXISTING:.*]] = arith.constant 42 : index
// CHECK: memref_ext.alloc_workspace() from %arg2 offset = [%[[EXISTING]]] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xf32, #hivm.address_space<gm>>
// CHECK-NOT: arith.constant 0 : index
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @already_has_offset(
      %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
      %arg1: memref<?xi8, #hivm.address_space<gm>>,
      %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}
  ) attributes {
      SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
      hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"
  } {
    %c42 = arith.constant 42 : index
    %0 = memref_ext.alloc_workspace() from %arg2 offset = [%c42]
        : from memref<?xi8, #hivm.address_space<gm>>
           to memref<64x64xf32, #hivm.address_space<gm>>
    return
  }
}

// -----

// CHECK-LABEL: func.func @no_workspaces
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: return
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @no_workspaces(
      %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
      %arg1: memref<?xi8, #hivm.address_space<gm>>
  ) attributes {
      hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"
  } {
    return
  }
}

// -----

// CHECK-LABEL: func.func @workspace_with_users
// CHECK: %[[OFF0:.*]] = arith.constant 0 : index
// CHECK: %[[WS:.*]] = memref_ext.alloc_workspace() from %arg2 offset = [%[[OFF0]]]
// CHECK: memref.subview %[[WS]]
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @workspace_with_users(
      %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
      %arg1: memref<?xi8, #hivm.address_space<gm>>,
      %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}
  ) attributes {
      SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
      hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"
  } {
    %0 = memref_ext.alloc_workspace() from %arg2
        : from memref<?xi8, #hivm.address_space<gm>>
           to memref<3x8x64x64xf32, #hivm.address_space<gm>>
    %idx = arith.constant 0 : index
    %sv = memref.subview %0[%idx, %idx, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1]
        : memref<3x8x64x64xf32, #hivm.address_space<gm>>
       to memref<1x1x64x64xf32, strided<[32768, 4096, 64, 1], offset: ?>,
                                 #hivm.address_space<gm>>
    return
  }
}
