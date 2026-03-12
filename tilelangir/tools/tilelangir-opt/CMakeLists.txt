get_tilelangir_link_libraries(LIBS)
add_tilelangir_install_tool(tilelangir-opt
  tilelangir-opt.cpp
  DEPENDS ${LIBS} tilelangir_transforms
  LINK_LIBS tilelangir_transforms ${LIBS}
)
target_link_libraries(tilelangir-opt PRIVATE tilelangir_headers)

if(TILELANGIR_MLIR_LIBS)
  target_link_libraries(tilelangir-opt PRIVATE
    -Wl,--start-group
    ${TILELANGIR_MLIR_LIBS}
    -Wl,--end-group
    ${TILELANGIR_SYSTEM_LIBS}
  )
endif()
