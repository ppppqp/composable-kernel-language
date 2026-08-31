execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${PYTHONPATH_VALUE}"
          "${Python3_EXECUTABLE}" "${INPUT_FILE}"
  RESULT_VARIABLE emit_result OUTPUT_VARIABLE source ERROR_VARIABLE emit_error)
if(NOT emit_result EQUAL 0)
  message(FATAL_ERROR "task tracing failed:\n${emit_error}")
endif()
if(NOT source MATCHES "\"ckl.task\"" OR NOT source MATCHES "\"ckl.invoke\"")
  message(FATAL_ERROR "trace did not construct generated task operations:\n${source}")
endif()
set(generated "${CMAKE_CURRENT_BINARY_DIR}/frontend-traced-task.mlir")
file(WRITE "${generated}" "${source}")
execute_process(
  COMMAND "${CKL_OPT}" "${generated}"
  RESULT_VARIABLE parse_result OUTPUT_VARIABLE output ERROR_VARIABLE parse_error)
if(NOT parse_result EQUAL 0)
  message(FATAL_ERROR "traced task failed dialect verification:\n${parse_error}\n${source}")
endif()
if(NOT output MATCHES "ckl.task @copy_task" OR NOT output MATCHES "ckl.invoke @copy_task")
  message(FATAL_ERROR "traced task did not survive round trip:\n${output}")
endif()
