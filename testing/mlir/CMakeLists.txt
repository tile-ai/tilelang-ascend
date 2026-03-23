get_property(TILELANGIR_TOOLS_DIRS GLOBAL PROPERTY TILELANGIR_TOOLS_DIRS)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TILELANGIR_TEST_DEPENDS
  FileCheck
  tilelangir-opt
)

add_lit_testsuite(check-tilelangir "Running the tilelangir regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TILELANGIR_TEST_DEPENDS}
)

set_target_properties(check-tilelangir PROPERTIES FOLDER "Tests")
