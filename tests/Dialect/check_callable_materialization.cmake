execute_process(
  COMMAND "${CKL_OPT}" "${INPUT_FILE}" --ckl-select-alternatives
          --ckl-materialize-selected-alternatives
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "callable alternative materialization failed:\n${error}")
endif()
foreach(expected
    "alternative = \"direct\""
    "alternative = \"permuted\""
    "implementation = \"direct_impl\""
    "implementation = \"permuted_impl\""
    "implementation_id = \"layout.direct\""
    "implementation_id = \"layout.permuted\""
    "arith.constant 11 : index"
    "arith.constant 22 : index")
  string(FIND "${output}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "materialized output is missing ${expected}:\n${output}")
  endif()
endforeach()
if(output MATCHES "ckl.invoke" OR output MATCHES "ckl.task" OR
   output MATCHES "func.call" OR output MATCHES "func.func private @.*_impl")
  message(FATAL_ERROR "resolved callable scaffolding survived inlining:\n${output}")
endif()
