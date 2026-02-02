set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS TilelangOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
add_mlir_doc(TilelangOps TilelangOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TilelangDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(TilelangDialect TilelangDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TilelangTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS TilelangInterfaces.td)
mlir_tablegen(AttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(AttrInterfaces.cpp.inc -gen-attr-interface-defs)

set(LLVM_TARGET_DEFINITIONS TilelangTypeInterfaces.td)
mlir_tablegen(TilelangTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TilelangTypeInterfaces.cpp.inc -gen-type-interface-defs)

#FIXME TODO: When upgrading to Tilelang 3.4.0, remove the commented line below 
set(LLVM_TARGET_DEFINITIONS TilelangOpInterfaces.td)
mlir_tablegen(OpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(TilelangTableGen)
