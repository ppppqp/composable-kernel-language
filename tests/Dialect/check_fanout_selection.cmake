execute_process(
  COMMAND "${CKL_OPT}" --ckl-select-alternatives "${INPUT_FILE}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "CKL fan-out selection failed: ${error}")
endif()
string(REGEX MATCHALL "ckl.compose" composed "${output}")
list(LENGTH composed compose_count)
if(NOT compose_count EQUAL 3)
  message(FATAL_ERROR "fan-out selection did not resolve all three graph edges")
endif()
foreach(expected
    "ckl.graph_score = 150"
    "ckl.graph_combinations_explored = 2"
    "implementation=producer.direct.v1"
    "ckl.consumer_alternative = \"direct\""
    "ckl.producer_alternative = \"direct\"")
  string(FIND "${output}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "fan-out selection output is missing: ${expected}")
  endif()
endforeach()

execute_process(
  COMMAND "${CKL_OPT}" "--ckl-select-alternatives=maximum-combinations=1" "${INPUT_FILE}"
  RESULT_VARIABLE bounded_result
  ERROR_VARIABLE bounded_error
)
if(bounded_result EQUAL 0)
  message(FATAL_ERROR "bounded fan-out selection unexpectedly committed an unproven choice")
endif()
foreach(expected
    "no proven optimal task-graph selection"
    "task graph search limit reached before proving optimality")
  string(FIND "${bounded_error}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "bounded fan-out diagnostic is missing: ${expected}")
  endif()
endforeach()
