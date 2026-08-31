execute_process(
  COMMAND "${CKL_OPT}" "${INPUT_FILE}" --ckl-select-alternatives
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "memory-boundary selection failed:\n${error}")
endif()
if(NOT output MATCHES "alternative = \"direct\".*ckl.graph_score = 0" OR
   NOT output MATCHES "alternative = \"permuted\".*ckl.graph_score = 0")
  message(FATAL_ERROR "fixed memory layouts did not select matching alternatives:\n${output}")
endif()
