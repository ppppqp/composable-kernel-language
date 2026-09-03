execute_process(
  COMMAND "${CKL_OPT}" "${INPUT_FILE}" --ckl-resolve-task-alternatives
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "fixed-point task resolution failed:\n${error}")
endif()
foreach(expected
    "alternative = \"composite\""
    "alternative = \"leaf\""
    "arith.constant 41 : index"
    "arith.constant 42 : index")
  string(FIND "${output}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "fixed-point output is missing ${expected}:\n${output}")
  endif()
endforeach()
if(output MATCHES "ckl.invoke" OR output MATCHES "ckl.task" OR
   output MATCHES "func.call" OR output MATCHES "func.func private")
  message(FATAL_ERROR "fixed-point task scaffolding survived:\n${output}")
endif()

execute_process(
  COMMAND "${CKL_OPT}" "${RECURSIVE_FILE}"
          "--ckl-resolve-task-alternatives=maximum-rounds=2"
  RESULT_VARIABLE recursive_result OUTPUT_VARIABLE recursive_output
  ERROR_VARIABLE recursive_error)
if(recursive_result EQUAL 0)
  message(FATAL_ERROR "recursive callable task unexpectedly resolved:\n${recursive_output}")
endif()
string(FIND "${recursive_error}" "may contain a recursive callable-alternative cycle"
       recursive_position)
if(recursive_position EQUAL -1)
  message(FATAL_ERROR "recursive task did not report the round-limit diagnostic:\n${recursive_error}")
endif()
