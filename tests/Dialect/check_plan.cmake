execute_process(
  COMMAND "${CKL_OPT}" --ckl-plan-compositions "${INPUT_FILE}"
  RESULT_VARIABLE plan_result
  OUTPUT_VARIABLE plan_output
  ERROR_VARIABLE plan_error
)
if(NOT plan_result EQUAL 0)
  message(FATAL_ERROR "CKL composition planning failed: ${plan_error}")
endif()
string(FIND "${plan_output}" "ckl.convert_layout" convert_position)
string(FIND "${plan_output}" "kind = \"local-permutation\"" kind_position)
if(convert_position EQUAL -1 OR kind_position EQUAL -1)
  message(FATAL_ERROR "planning did not produce the expected local-permutation conversion")
endif()
