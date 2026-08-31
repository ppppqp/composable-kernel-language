execute_process(
  COMMAND "${CKL_OPT}" --ckl-plan-compositions --ckl-schedule-conversions "${INPUT_FILE}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "CKL conversion-class scheduling failed: ${error}")
endif()
foreach(expected
    "ckl.local_permute"
    "ckl.subgroup_exchange"
    "ckl.shared_store"
    "ckl.workgroup_barrier"
    "ckl.shared_load"
    "ckl.global_store"
    "ckl.kernel_boundary"
    "ckl.global_load")
  string(FIND "${output}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "conversion-class output is missing: ${expected}")
  endif()
endforeach()
string(FIND "${output}" "ckl.shared_store" shared_store)
string(FIND "${output}" "ckl.workgroup_barrier" shared_barrier)
string(FIND "${output}" "ckl.shared_load" shared_load)
string(FIND "${output}" "ckl.global_store" global_store)
string(FIND "${output}" "ckl.kernel_boundary" kernel_boundary)
string(FIND "${output}" "ckl.global_load" global_load)
if(NOT shared_store LESS shared_barrier OR NOT shared_barrier LESS shared_load OR
   NOT global_store LESS kernel_boundary OR NOT kernel_boundary LESS global_load)
  message(FATAL_ERROR "shared/global conversion phases are not ordered correctly")
endif()
foreach(unexpected "ckl.compose" "ckl.convert_layout" "ckl.shared_exchange"
                   "ckl.global_exchange" "kind = \"identity\"")
  string(FIND "${output}" "${unexpected}" position)
  if(NOT position EQUAL -1)
    message(FATAL_ERROR "conversion-class scheduling left unresolved IR: ${unexpected}")
  endif()
endforeach()
