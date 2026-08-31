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
  NAME ckl-dialect-dsl-roundtrip
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/dsl.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_roundtrip.cmake
)
add_test(
  NAME ckl-dialect-tile-memory-roundtrip
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/tile_memory.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_roundtrip.cmake
)
add_test(
  NAME ckl-dialect-schedule-conversions
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/roundtrip.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_schedule.cmake
)
add_test(
  NAME ckl-dialect-conversion-classes
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/conversion_classes.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_conversion_classes.cmake
)
add_test(
  NAME ckl-dialect-select-alternatives
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/task_selection.mlir
    -DREJECTED_FILE=${CMAKE_CURRENT_BINARY_DIR}/task_selection_rejected.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_selection.cmake
)
add_test(
  NAME ckl-dialect-select-linear-pipeline
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/linear_pipeline.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_linear_pipeline.cmake
)
add_test(
  NAME ckl-dialect-select-fanout
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/fanout_selection.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_fanout_selection.cmake
)
add_test(
  NAME ckl-dialect-invoke-graph
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/invoke_graph.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_invoke_graph.cmake
)
add_test(
  NAME ckl-dialect-nvidia-alternatives
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/nvidia_alternatives.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_nvidia_alternatives.cmake
)
add_test(
  NAME ckl-dialect-nvidia-materialize
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/nvidia_materialize.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_nvidia_materialize.cmake
)
add_test(
  NAME ckl-dialect-nvidia-lower-mma
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/nvidia_materialize.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_nvidia_lower_mma.cmake
)
add_test(
  NAME ckl-dialect-nvidia-fragment-io
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/nvidia_fragment_io.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_nvidia_fragment_io.cmake
)
add_test(
  NAME ckl-dialect-invalid-diagnostics
  COMMAND $<TARGET_FILE:ckl-opt> --verify-diagnostics ${CMAKE_CURRENT_SOURCE_DIR}/invalid.mlir
)
add_test(
  NAME ckl-dialect-invalid-dsl-diagnostics
  COMMAND $<TARGET_FILE:ckl-opt> --split-input-file --verify-diagnostics ${CMAKE_CURRENT_SOURCE_DIR}/invalid_dsl.mlir
)
