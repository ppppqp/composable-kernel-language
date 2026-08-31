execute_process(
  COMMAND "${CKL_OPT}" "${INPUT_FILE}" --ckl-select-alternatives
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "isolated alternative selection failed:\n${error}")
endif()
if(NOT output MATCHES "alternative = \"fast\"" OR
   NOT output MATCHES "ckl.implementation_id = \"choose.fast\"" OR
   NOT output MATCHES "ckl.graph_score = 2")
  message(FATAL_ERROR "isolated invocation did not select the lowest-cost alternative:\n${output}")
endif()
