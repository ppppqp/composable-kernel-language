add_test(
  NAME ckl-dialect-roundtrip
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/roundtrip.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_roundtrip.cmake
)
add_test(
  NAME ckl-dialect-plan-compositions
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/roundtrip.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_plan.cmake
)
add_test(
  NAME ckl-dialect-invalid-diagnostics
  COMMAND $<TARGET_FILE:ckl-opt> --verify-diagnostics ${CMAKE_CURRENT_SOURCE_DIR}/invalid.mlir
)
