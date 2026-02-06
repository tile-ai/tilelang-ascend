tilelangir_tablegen(tilelangir_transforms_incgen
  TD_FILES Passes.td
  ARGS -gen-pass-decls -name TileLangIR
)
set(TILELANGIR_TRANSFORMS_GEN_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/../.. CACHE PATH "Generated Transforms headers (include root)")
