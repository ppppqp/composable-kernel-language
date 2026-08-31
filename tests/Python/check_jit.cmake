execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${PYTHONPATH_VALUE}" "CKL_OPT=${CKL_OPT}"
          "${Python3_EXECUTABLE}" "${INPUT_FILE}"
  RESULT_VARIABLE compile_result OUTPUT_VARIABLE output ERROR_VARIABLE compile_error)
if(NOT compile_result EQUAL 0)
  message(FATAL_ERROR "@ckl.jit compilation failed:\n${compile_error}")
endif()
if(NOT output MATCHES "ckl.task @copy_task" OR NOT output MATCHES "ckl.invoke @copy_task")
  message(FATAL_ERROR "compiled JIT module is missing task operations:\n${output}")
endif()
