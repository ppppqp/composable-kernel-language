add_test(
  NAME ckl-dialect-roundtrip
  COMMAND $<TARGET_FILE:ckl-opt> ${CMAKE_CURRENT_SOURCE_DIR}/roundtrip.mlir
)
add_test(
  NAME ckl-dialect-plan-compositions
  COMMAND $<TARGET_FILE:ckl-opt> --ckl-plan-compositions ${CMAKE_CURRENT_SOURCE_DIR}/roundtrip.mlir
)
add_test(
  NAME ckl-dialect-invalid-diagnostics
  COMMAND $<TARGET_FILE:ckl-opt> --verify-diagnostics ${CMAKE_CURRENT_SOURCE_DIR}/invalid.mlir
)
