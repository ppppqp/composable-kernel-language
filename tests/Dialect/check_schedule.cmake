execute_process(
  COMMAND "${CKL_OPT}" --ckl-plan-compositions --ckl-schedule-conversions "${INPUT_FILE}"
  RESULT_VARIABLE schedule_result
  OUTPUT_VARIABLE schedule_output
  ERROR_VARIABLE schedule_error
)
if(NOT schedule_result EQUAL 0)
  message(FATAL_ERROR "CKL conversion scheduling failed: ${schedule_error}")
endif()
string(FIND "${schedule_output}" "ckl.local_permute" scheduled_position)
string(FIND "${schedule_output}" "move_count = 4" provenance_position)
string(FIND "${schedule_output}" "reason = \"ownership agrees but local slots differ\"" reason_position)
string(FIND "${schedule_output}" "ckl.convert_layout" semantic_position)
if(scheduled_position EQUAL -1 OR provenance_position EQUAL -1 OR
   reason_position EQUAL -1 OR NOT semantic_position EQUAL -1)
  message(FATAL_ERROR "scheduling did not replace the semantic conversion or preserve provenance")
endif()
