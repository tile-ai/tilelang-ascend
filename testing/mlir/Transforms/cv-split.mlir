// RUN: tilelangir-opt --tilelangir-cv-split %s | FileCheck %s

// CHECK-LABEL: @cv_split
// CHECK: %[[EMPTY:.*]] = tensor.empty
// CHECK: %[[RESULT:.*]] = hivm.hir.vmul
// CHECK-SAME: outs(%[[EMPTY]] :
// CHECK: return %[[RESULT]] :
func.func @cv_split(%arg0: tensor<3x3x64xf32>, %arg1: tensor<3x3x64xf32>) -> tensor<3x3x64xf32> {
    %0 = tensor.empty(): tensor<3x3x64xf32>
    %1 = hivm.hir.vmul ins(%arg0, %arg1: tensor<3x3x64xf32>, tensor<3x3x64xf32>) outs(%0: tensor<3x3x64xf32>) -> tensor<3x3x64xf32>
    func.return %1: tensor<3x3x64xf32>
}
