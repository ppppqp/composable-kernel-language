execute_process(
  COMMAND "${CKL_OPT}" "${INPUT_FILE}" --ckl-select-alternatives
          --ckl-materialize-selected-alternatives
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "callable alternative materialization failed:\n${error}")
endif()
foreach(expected
    "call @direct_impl"
    "call @permuted_impl"
    "ckl.selected_alternative = \"direct\""
    "ckl.selected_alternative = \"permuted\""
    "ckl.implementation_id = \"layout.direct\""
    "ckl.implementation_id = \"layout.permuted\"")
  string(FIND "${output}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "materialized output is missing ${expected}:\n${output}")
  endif()
endforeach()
if(output MATCHES "ckl.invoke" OR output MATCHES "ckl.task")
  message(FATAL_ERROR "resolved CKL task operations survived materialization:\n${output}")
endif()
