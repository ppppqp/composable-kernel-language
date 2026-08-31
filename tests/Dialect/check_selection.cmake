execute_process(
  COMMAND "${CKL_OPT}" --ckl-select-alternatives "${INPUT_FILE}"
  RESULT_VARIABLE selection_result
  OUTPUT_VARIABLE selection_output
  ERROR_VARIABLE selection_error
)
if(NOT selection_result EQUAL 0)
  message(FATAL_ERROR "CKL task selection failed: ${selection_error}")
endif()
foreach(expected
    "ckl.compose"
    "ckl.producer_alternative = \"direct\""
    "ckl.consumer_alternative = \"operand\""
    "ckl.producer_implementation_id = \"producer.direct.v1\""
    "ckl.producer_implementation = @producer_direct"
    "ckl.consumer_implementation = @consumer_operand"
    "score = 35"
    "explanation = \"resource limit exceeded\"")
  string(FIND "${selection_output}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "task selection output is missing: ${expected}")
  endif()
endforeach()

execute_process(
  COMMAND "${CKL_OPT}" --ckl-select-alternatives --ckl-plan-compositions "${INPUT_FILE}"
  RESULT_VARIABLE pipeline_result
  OUTPUT_VARIABLE pipeline_output
  ERROR_VARIABLE pipeline_error
)
if(NOT pipeline_result EQUAL 0)
  message(FATAL_ERROR "CKL selection/planning pipeline failed: ${pipeline_error}")
endif()
foreach(expected
    "ckl.convert_layout"
    "ckl.producer_alternative = \"direct\""
    "ckl.producer_implementation = @producer_direct"
    "ckl.considered_alternatives")
  string(FIND "${pipeline_output}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "selection provenance did not survive planning: ${expected}")
  endif()
endforeach()

file(READ "${INPUT_FILE}" rejected_input)
string(REPLACE "capabilities = [\"mma\"]\n      attributes"
               "capabilities = []\n      attributes"
               rejected_input "${rejected_input}")
file(WRITE "${REJECTED_FILE}" "${rejected_input}")
execute_process(
  COMMAND "${CKL_OPT}" --ckl-select-alternatives "${REJECTED_FILE}"
  RESULT_VARIABLE rejected_result
  ERROR_VARIABLE rejected_error
)
if(rejected_result EQUAL 0)
  message(FATAL_ERROR "CKL task selection unexpectedly accepted an unavailable capability")
endif()
foreach(expected "no legal task-alternative pair" "target capability unavailable")
  string(FIND "${rejected_error}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "failed selection diagnostic is missing: ${expected}")
  endif()
endforeach()
