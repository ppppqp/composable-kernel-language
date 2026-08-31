execute_process(
  COMMAND "${CKL_OPT}" "${INPUT_FILE}" --ckl-nvidia-materialize-mma-sync
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "ckl-opt failed:\n${error}")
endif()
if(NOT output MATCHES "ckl_nvidia.mma_sync")
  message(FATAL_ERROR "selected NVIDIA invocation was not materialized:\n${output}")
endif()
if(output MATCHES "ckl.invoke @mma")
  message(FATAL_ERROR "selected invocation survived materialization:\n${output}")
endif()
if(NOT output MATCHES "ckl.graph_score = 1")
  message(FATAL_ERROR "selection provenance was not preserved:\n${output}")
endif()
if(NOT output MATCHES "ckl.implementation_id = \"nvidia.mma.sync.m16n8k16.row.col.f32.f16.f16.f32\"")
  message(FATAL_ERROR "implementation identity was not preserved:\n${output}")
endif()

set(materialized_file "${CMAKE_CURRENT_BINARY_DIR}/nvidia_materialized.mlir")
file(WRITE "${materialized_file}" "${output}")
execute_process(
  COMMAND "${CKL_OPT}" "${materialized_file}"
  RESULT_VARIABLE roundtrip_result ERROR_VARIABLE roundtrip_error)
if(NOT roundtrip_result EQUAL 0)
  message(FATAL_ERROR "materialized NVIDIA IR did not round-trip:\n${roundtrip_error}")
endif()
