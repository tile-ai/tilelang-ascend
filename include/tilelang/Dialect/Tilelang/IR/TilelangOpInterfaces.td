#ifndef TILELANG_OP_INTERFACES
#define TILELANG_OP_INTERFACES

include "mlir/IR/OpBase.td"


def TT_DescriptorOpInterface : OpInterface<"DescriptorOpInterface"> {
  let description = [{
    Common interface to get the descriptor argument from an operation on tensor descriptors.
  }];

  let cppNamespace = "::mlir::tilelang";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the descriptor",
      /*retType=*/"::mlir::TypedValue<mlir::tilelang::TensorDescType>",
      /*methodName=*/"getDesc",
      /*args=*/(ins)>,
  ];
}

def TT_DescriptorStoreLikeOpInterface : OpInterface<"DescriptorStoreLikeOpInterface", [TT_DescriptorOpInterface]> {
  let cppNamespace = "::mlir::tilelang";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get Source tensor",
      /*retType=*/"::mlir::TypedValue<mlir::RankedTensorType>",
      /*methodName=*/"getSrc",
      /*args=*/(ins)>,
  ];
}


#endif // TILELANG_OP_INTERFACES
