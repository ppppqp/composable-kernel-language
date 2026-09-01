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
  NAME ckl-dialect-select-isolated-invoke
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/isolated_selection.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_isolated_selection.cmake
)
add_test(
  NAME ckl-dialect-select-memory-boundaries
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/memory_boundary_selection.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_memory_boundary_selection.cmake
)
add_test(
  NAME ckl-dialect-materialize-callable-alternatives
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/callable_materialization.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_callable_materialization.cmake
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
  NAME ckl-dialect-nvidia-nvvm
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/nvidia_fragment_io.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_nvidia_nvvm.cmake
)
if(CKL_ENABLE_CUDA_TESTS)
  if(NOT CKL_CUDA_TOOLKIT_ROOT OR NOT CKL_CUDA_ARCH OR NOT CKL_CUDA_PTX_FEATURE)
    message(FATAL_ERROR
      "CUDA tests require CKL_CUDA_TOOLKIT_ROOT, CKL_CUDA_ARCH, and CKL_CUDA_PTX_FEATURE")
  endif()
  if(NOT EXISTS "${CKL_CUDA_TOOLKIT_ROOT}/bin/ptxas")
    message(FATAL_ERROR "ptxas not found under CKL_CUDA_TOOLKIT_ROOT")
  endif()
  if(NOT EXISTS "${CKL_CUDA_TOOLKIT_ROOT}/nvvm/libdevice/libdevice.10.bc")
    message(FATAL_ERROR "libdevice.10.bc not found under CKL_CUDA_TOOLKIT_ROOT")
  endif()
  find_path(CKL_CUDA_DRIVER_INCLUDE_DIR cuda.h REQUIRED)
  find_library(CKL_CUDA_DRIVER_LIBRARY cuda REQUIRED)
  add_executable(ckl-nvidia-runtime-validation NvidiaRuntimeValidation.cpp)
  target_include_directories(ckl-nvidia-runtime-validation PRIVATE
    ${CKL_CUDA_DRIVER_INCLUDE_DIR})
  target_link_libraries(ckl-nvidia-runtime-validation PRIVATE
    MLIRGPUDialect
    MLIRParser
    MLIRRegisterAllDialects
    ${CKL_CUDA_DRIVER_LIBRARY})
  add_test(
    NAME ckl-dialect-nvidia-cubin
    COMMAND ${CMAKE_COMMAND}
      -DCKL_OPT=$<TARGET_FILE:ckl-opt>
      -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/nvidia_fragment_io.mlir
      -DCUDA_ROOT=${CKL_CUDA_TOOLKIT_ROOT}
      -DCUDA_ARCH=${CKL_CUDA_ARCH}
      -DCUDA_PTX_FEATURE=${CKL_CUDA_PTX_FEATURE}
      -DCKL_CUDA_RUNNER=$<TARGET_FILE:ckl-nvidia-runtime-validation>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_nvidia_cubin.cmake
  )
  set_tests_properties(ckl-dialect-nvidia-cubin PROPERTIES LABELS "cuda")
endif()
add_test(
  NAME ckl-dialect-invalid-diagnostics
  COMMAND $<TARGET_FILE:ckl-opt> --verify-diagnostics ${CMAKE_CURRENT_SOURCE_DIR}/invalid.mlir
)
add_test(
  NAME ckl-dialect-invalid-dsl-diagnostics
  COMMAND $<TARGET_FILE:ckl-opt> --split-input-file --verify-diagnostics ${CMAKE_CURRENT_SOURCE_DIR}/invalid_dsl.mlir
)
