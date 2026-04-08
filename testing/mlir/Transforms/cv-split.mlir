// RUN: tilelangir-opt --tilelangir-cv-split %s | FileCheck %s

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16>, %arg4: memref<?xf16>, %arg5: memref<?xf16>, %arg6: memref<?xf16>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, hivm.part_of_mix, mix_mode = "mix"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c128_i32 = arith.constant 128 : i32
    %1 = arith.muli %c128_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [512, 128], strides: [%2, %0] : memref<?xf16> to memref<512x128xf16, strided<[128, 1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [512, 128], strides: [%2, %0] : memref<?xf16> to memref<512x128xf16, strided<[128, 1]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [512, 128], strides: [%2, %0] : memref<?xf16> to memref<512x128xf16, strided<[128, 1]>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [512, 128], strides: [%2, %0] : memref<?xf16> to memref<512x128xf16, strided<[128, 1]>>
    %3 = hivm.hir.get_block_idx -> i64
    %4 = arith.trunci %3 : i64 to i32
    %5 = memref_ext.alloc_workspace() : memref<64x64xf32>
    %6 = memref_ext.alloc_workspace() : memref<64x64xf16>
    %7 = memref_ext.alloc_workspace() : memref<64x128xf32>
    %8 = tensor.empty() : tensor<64x128xf16>
    %9 = tensor.empty() : tensor<64x1xf32>
    %10 = tensor.empty() : tensor<64x1xf32>
    %11 = tensor.empty() : tensor<64x128xf32>
    %12 = tensor.empty() : tensor<64x64xf32>
    %13 = tensor.empty() : tensor<64x128xf16>
    %14 = tensor.empty() : tensor<64x128xf32>
    %c64_i32 = arith.constant 64 : i32
    %15 = arith.muli %4, %c64_i32 : i32
    %16 = arith.index_cast %15 : i32 to index
    %subview = memref.subview %reinterpret_cast[%16, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>> to memref<64x128xf16, strided<[128, 1], offset: ?>>
    %alloc = memref.alloc() : memref<64x128xf16>
    memref.copy %subview, %alloc : memref<64x128xf16, strided<[128, 1], offset: ?>> to memref<64x128xf16>
    %17 = bufferization.to_tensor %alloc restrict : memref<64x128xf16>
    %inserted_slice = tensor.insert_slice %17 into %8[0, 0] [64, 128] [1, 1] : tensor<64x128xf16> into tensor<64x128xf16>
    %c0_i32 = arith.constant 0 : i32
    %18 = arith.sitofp %c0_i32 : i32 to f32
    %19 = hivm.hir.vbrc ins(%18 : f32) outs(%11 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %alloc_3 = memref.alloc() : memref<64x128xf32>
    bufferization.materialize_in_destination %19 in writable %alloc_3 : (tensor<64x128xf32>, memref<64x128xf32>) -> ()
    hivm.hir.store ins(%alloc_3 : memref<64x128xf32>) outs(%7 : memref<64x128xf32>)
    %20 = arith.sitofp %c0_i32 : i32 to f32
    %21 = hivm.hir.vbrc ins(%20 : f32) outs(%10 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %cst = arith.constant 0xFF800000 : f32
    %22 = hivm.hir.vbrc ins(%cst : f32) outs(%9 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %cst_4 = arith.constant 0.0883883461 : f32
    %23 = hivm.hir.vbrc ins(%cst_4 : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %c8_i32 = arith.constant 8 : i32
    %c1_i32_5 = arith.constant 1 : i32
    %alloc_10 = memref.alloc() : memref<64x128xf16>
    %alloc_21 = memref.alloc() : memref<64x64xf16>
    %24:6 = scf.for %arg13 = %c0_i32 to %c8_i32 step %c1_i32_5 iter_args(%arg14 = %inserted_slice, %arg15 = %19, %arg16 = %23, %arg17 = %22, %arg18 = %21, %arg19 = %14) -> (tensor<64x128xf16>, tensor<64x128xf32>, tensor<64x64xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x128xf32>)  : i32 {
      %28 = tensor.empty() : tensor<64x128xf16>
      %29 = tensor.empty() : tensor<64x128xf16>
      %30 = tensor.empty() : tensor<64x64xf32>
      %31 = tensor.empty() : tensor<64x64xf32>
      %32 = tensor.empty() : tensor<64x64xf16>
      %33 = tensor.empty() : tensor<64x64xf16>
      %34 = tensor.empty() : tensor<64x1xf32>
      %35 = tensor.empty() : tensor<64x1xf32>
      %36 = tensor.empty() : tensor<64x1xf32>
      %37 = tensor.empty() : tensor<64x64xf32>
      %38 = tensor.empty() : tensor<64x1xf32>
      %39 = tensor.empty() : tensor<64x1xf32>
      %c64_i32_8 = arith.constant 64 : i32
      %40 = arith.muli %arg13, %c64_i32_8 : i32
      %41 = arith.index_cast %40 : i32 to index
      %subview_9 = memref.subview %reinterpret_cast_1[%41, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>> to memref<64x128xf16, strided<[128, 1], offset: ?>>
      memref.copy %subview_9, %alloc_10 : memref<64x128xf16, strided<[128, 1], offset: ?>> to memref<64x128xf16>
      %42 = bufferization.to_tensor %alloc_10 restrict : memref<64x128xf16>
      %inserted_slice_11 = tensor.insert_slice %42 into %28[0, 0] [64, 128] [1, 1] : tensor<64x128xf16> into tensor<64x128xf16>
      %alloc_12 = memref.alloc() : memref<64x64xf32>
      hivm.hir.load ins(%5 : memref<64x64xf32>) outs(%alloc_12 : memref<64x64xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %43 = bufferization.to_tensor %alloc_12 restrict : memref<64x64xf32>
      %true = arith.constant true
      %44 = arith.index_cast %c64_i32_8 : i32 to index
      %c128_i32_13 = arith.constant 128 : i32
      %45 = arith.index_cast %c128_i32_13 : i32 to index
      %46 = hivm.hir.mmadL1 {b_transpose} ins(%arg14, %inserted_slice_11, %true, %44, %44, %45 : tensor<64x128xf16>, tensor<64x128xf16>, i1, index, index, index) outs(%43 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %alloc_14 = memref.alloc() : memref<64x64xf32>
      bufferization.materialize_in_destination %46 in writable %alloc_14 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
      hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_14 : memref<64x64xf32>) outs(%5 : memref<64x64xf32>)
      %alloc_15 = memref.alloc() : memref<64x128xf32>
      hivm.hir.load ins(%7 : memref<64x128xf32>) outs(%alloc_15 : memref<64x128xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %47 = bufferization.to_tensor %alloc_15 restrict : memref<64x128xf32>
      %alloc_16 = memref.alloc() : memref<64x64xf32>
      hivm.hir.load ins(%5 : memref<64x64xf32>) outs(%alloc_16 : memref<64x64xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %48 = bufferization.to_tensor %alloc_16 restrict : memref<64x64xf32>
      %49 = hivm.hir.vmul ins(%48, %arg16 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%48 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %50 = hivm.hir.vreduce <max> ins(%49 : tensor<64x64xf32>) outs(%35 : tensor<64x1xf32>) reduce_dims = [1] -> tensor<64x1xf32>
      %inserted_slice_17 = tensor.insert_slice %50 into %35[0, 0] [64, 1] [1, 1] : tensor<64x1xf32> into tensor<64x1xf32>
      %51 = hivm.hir.vmax ins(%arg17, %inserted_slice_17 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%39 : tensor<64x1xf32>) -> tensor<64x1xf32>
      %52 = hivm.hir.vsub ins(%arg17, %51 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%38 : tensor<64x1xf32>) -> tensor<64x1xf32>
      %53 = hivm.hir.vexp ins(%52 : tensor<64x1xf32>) outs(%34 : tensor<64x1xf32>) -> tensor<64x1xf32>
      %54 = hivm.hir.vsub ins(%49, %51 : tensor<64x64xf32>, tensor<64x1xf32>) outs(%37 : tensor<64x64xf32>) broadcast = [1] -> tensor<64x64xf32>
      %55 = hivm.hir.vexp ins(%54 : tensor<64x64xf32>) outs(%49 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %56 = hivm.hir.vreduce <sum> ins(%55 : tensor<64x64xf32>) outs(%36 : tensor<64x1xf32>) reduce_dims = [1] -> tensor<64x1xf32>
      %inserted_slice_18 = tensor.insert_slice %56 into %36[0, 0] [64, 1] [1, 1] : tensor<64x1xf32> into tensor<64x1xf32>
      %57 = hivm.hir.vmul ins(%arg18, %53 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%arg18 : tensor<64x1xf32>) -> tensor<64x1xf32>
      %58 = hivm.hir.vadd ins(%57, %inserted_slice_18 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%57 : tensor<64x1xf32>) -> tensor<64x1xf32>
      %59 = hivm.hir.vmul ins(%47, %53 : tensor<64x128xf32>, tensor<64x1xf32>) outs(%47 : tensor<64x128xf32>) broadcast = [1] -> tensor<64x128xf32>
      %60 = hivm.hir.vcast ins(%55 : tensor<64x64xf32>) outs(%32 : tensor<64x64xf16>) -> tensor<64x64xf16>
      %alloc_19 = memref.alloc() : memref<64x64xf32>
      bufferization.materialize_in_destination %55 in writable %alloc_19 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
      hivm.hir.store ins(%alloc_19 : memref<64x64xf32>) outs(%5 : memref<64x64xf32>)
      %alloc_20 = memref.alloc() : memref<64x128xf32>
      bufferization.materialize_in_destination %59 in writable %alloc_20 : (tensor<64x128xf32>, memref<64x128xf32>) -> ()
      hivm.hir.store ins(%alloc_20 : memref<64x128xf32>) outs(%7 : memref<64x128xf32>)
      bufferization.materialize_in_destination %60 in writable %alloc_21 : (tensor<64x64xf16>, memref<64x64xf16>) -> ()
      hivm.hir.store ins(%alloc_21 : memref<64x64xf16>) outs(%6 : memref<64x64xf16>)
      %c0_i32_22 = arith.constant 0 : i32
      %61 = arith.sitofp %c0_i32_22 : i32 to f32
      %62 = hivm.hir.vbrc ins(%61 : f32) outs(%52 : tensor<64x1xf32>) -> tensor<64x1xf32>
      %63 = hivm.hir.vadd ins(%62, %51 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%arg17 : tensor<64x1xf32>) -> tensor<64x1xf32>
      %subview_23 = memref.subview %reinterpret_cast_0[%41, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>> to memref<64x128xf16, strided<[128, 1], offset: ?>>
      %alloc_24 = memref.alloc() : memref<64x128xf16>
      memref.copy %subview_23, %alloc_24 : memref<64x128xf16, strided<[128, 1], offset: ?>> to memref<64x128xf16>
      %64 = bufferization.to_tensor %alloc_24 restrict : memref<64x128xf16>
      %inserted_slice_25 = tensor.insert_slice %64 into %29[0, 0] [64, 128] [1, 1] : tensor<64x128xf16> into tensor<64x128xf16>
      %alloc_26 = memref.alloc() : memref<64x64xf16>
      hivm.hir.load ins(%6 : memref<64x64xf16>) outs(%alloc_26 : memref<64x64xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %65 = bufferization.to_tensor %alloc_26 restrict : memref<64x64xf16>
      %alloc_27 = memref.alloc() : memref<64x128xf32>
      hivm.hir.load ins(%7 : memref<64x128xf32>) outs(%alloc_27 : memref<64x128xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %66 = bufferization.to_tensor %alloc_27 restrict : memref<64x128xf32>
      %false = arith.constant false
      %67 = hivm.hir.mmadL1 ins(%65, %inserted_slice_25, %false, %44, %44, %45 : tensor<64x64xf16>, tensor<64x128xf16>, i1, index, index, index) outs(%66 : tensor<64x128xf32>) -> tensor<64x128xf32>
      %alloc_28 = memref.alloc() : memref<64x128xf32>
      bufferization.materialize_in_destination %67 in writable %alloc_28 : (tensor<64x128xf32>, memref<64x128xf32>) -> ()
      hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_28 : memref<64x128xf32>) outs(%7 : memref<64x128xf32>)
      scf.yield %arg14, %59, %arg16, %63, %58, %67 : tensor<64x128xf16>, tensor<64x128xf32>, tensor<64x64xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x128xf32>
    }
    %alloc_6 = memref.alloc() : memref<64x128xf32>
    hivm.hir.load ins(%7 : memref<64x128xf32>) outs(%alloc_6 : memref<64x128xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %25 = bufferization.to_tensor %alloc_6 restrict : memref<64x128xf32>
    %26 = hivm.hir.vdiv ins(%25, %24#4 : tensor<64x128xf32>, tensor<64x1xf32>) outs(%25 : tensor<64x128xf32>) broadcast = [1] -> tensor<64x128xf32>
    %27 = hivm.hir.vcast ins(%26 : tensor<64x128xf32>) outs(%13 : tensor<64x128xf16>) -> tensor<64x128xf16>
    %subview_7 = memref.subview %reinterpret_cast_2[%16, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>> to memref<64x128xf16, strided<[128, 1], offset: ?>>
    bufferization.materialize_in_destination %27 in writable %subview_7 : (tensor<64x128xf16>, memref<64x128xf16, strided<[128, 1], offset: ?>>) -> ()
    return
  }
}

// CHECK-LABEL:   func.func @flash_attention
// CHECK:           %[[VAL_24:.*]] = memref_ext.alloc_workspace() : memref<64x64xf32>
// CHECK:           %[[VAL_25:.*]] = memref_ext.alloc_workspace() : memref<64x64xf16>
// CHECK:           %[[VAL_26:.*]] = memref_ext.alloc_workspace() : memref<64x128xf32>
// CHECK:           scf.for
// CHECK:             scope.scope : () -> () {
// CHECK:               hivm.hir.load ins(%[[VAL_24]] :
// CHECK:               hivm.hir.mmadL1
// CHECK:               hivm.hir.fixpipe {enable_nz2nd}
// CHECK-SAME:                  outs(%[[VAL_24]] :
// CHECK:               scope.return
// CHECK:             } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
// CHECK:             %[[VAL_126:.*]]:3 = scope.scope : () -> (tensor<64x1xf32>, tensor<64x128xf32>, tensor<64x1xf32>) {
// CHECK:               hivm.hir.load ins(%[[VAL_26]] :
// CHECK:               hivm.hir.load ins(%[[VAL_24]] :
// CHECK:               hivm.hir.vmul
// CHECK:               hivm.hir.vreduce <max>
// CHECK:               hivm.hir.vmax
// CHECK:               hivm.hir.vsub
// CHECK:               hivm.hir.vexp
// CHECK:               hivm.hir.vsub
// CHECK:               hivm.hir.vexp
// CHECK:               hivm.hir.vreduce <sum>
// CHECK:               hivm.hir.vmul
// CHECK:               %[[VAL_106:.*]] = hivm.hir.vadd
// CHECK:               %[[VAL_107:.*]] = hivm.hir.vmul
// CHECK:               hivm.hir.vcast
// CHECK:               hivm.hir.store
// CHECK-SAME:                  outs(%[[VAL_24]] :
// CHECK:               hivm.hir.store
// CHECK-SAME:                  outs(%[[VAL_26]] :
// CHECK:               hivm.hir.store
// CHECK-SAME:                  outs(%[[VAL_25]] :
// CHECK:               hivm.hir.vbrc
// CHECK:               %[[VAL_113:.*]] = hivm.hir.vadd
// CHECK:               scope.return %[[VAL_106]], %[[VAL_107]], %[[VAL_113]] :
// CHECK:             } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK:             %[[VAL_114:.*]] = scope.scope : () -> tensor<64x128xf32> {
// CHECK:               hivm.hir.load ins(%[[VAL_25]] :
// CHECK:               hivm.hir.load ins(%[[VAL_26]] :
// CHECK:               %[[VAL_124:.*]] = hivm.hir.mmadL1
// CHECK:               hivm.hir.fixpipe {enable_nz2nd}
// CHECK-SAME:                  outs(%[[VAL_26]] :
// CHECK:               scope.return %[[VAL_124]] :
// CHECK:             } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
// CHECK:             scf.yield
// CHECK-SAME:          %[[VAL_126]]#1
// CHECK-SAME:          %[[VAL_126]]#2, %[[VAL_126]]#0, %[[VAL_114]] :

