execute_process(
  COMMAND "${CKL_OPT}" --ckl-select-linear-pipelines "${INPUT_FILE}"
  RESULT_VARIABLE pipeline_result
  OUTPUT_VARIABLE pipeline_output
  ERROR_VARIABLE pipeline_error
)
if(NOT pipeline_result EQUAL 0)
  message(FATAL_ERROR "CKL linear pipeline selection failed: ${pipeline_error}")
endif()
string(REGEX MATCHALL "ckl.compose" composed "${pipeline_output}")
list(LENGTH composed compose_count)
if(NOT compose_count EQUAL 2)
  message(FATAL_ERROR "linear selection did not resolve both task boundaries")
endif()
foreach(expected
    "ckl.pipeline_score = 10"
    "ckl.pipeline_stage = 0"
    "ckl.pipeline_stage = 1"
    "ckl.consumer_alternative = \"global\""
    "ckl.producer_alternative = \"global\""
    "implementation=middle.global.v1")
  string(FIND "${pipeline_output}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "linear pipeline output is missing: ${expected}")
  endif()
endforeach()

execute_process(
  COMMAND "${CKL_OPT}" --ckl-select-linear-pipelines --ckl-plan-compositions "${INPUT_FILE}"
  RESULT_VARIABLE planned_result
  OUTPUT_VARIABLE planned_output
  ERROR_VARIABLE planned_error
)
if(NOT planned_result EQUAL 0)
  message(FATAL_ERROR "CKL linear selection/planning failed: ${planned_error}")
endif()
foreach(expected "ckl.convert_layout" "ckl.pipeline_score = 10" "ckl.pipeline_provenance")
  string(FIND "${planned_output}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "linear pipeline provenance did not survive planning: ${expected}")
  endif()
endforeach()
