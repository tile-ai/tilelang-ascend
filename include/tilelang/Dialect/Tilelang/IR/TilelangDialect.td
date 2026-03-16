#ifndef TILELANG_DIALECT
#define TILELANG_DIALECT

include "mlir/IR/OpBase.td"

def Tilelang_Dialect : Dialect {
  let name = "tt";

  let cppNamespace = "::mlir::tilelang";

  let summary = "The Tilelang IR in MLIR";

  let description = [{
    Tilelang Dialect.

    Dependent Dialects:
      * Arith:
        * addf, addi, andi, cmpf, cmpi, divf, fptosi, ...
      * Math:
        * exp, sin, cos, log, ...
      * StructuredControlFlow:
        * for, if, while, yield, condition
      * ControlFlow:
        * br, cond_br
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "math::MathDialect",
    "scf::SCFDialect",
    "cf::ControlFlowDialect",
    "ub::UBDialect"
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

include "tilelang/Dialect/Tilelang/IR/TilelangTypes.td"


#endif // TILELANG_DIALECT
