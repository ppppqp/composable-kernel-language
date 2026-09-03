execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "PYTHONPATH=${PYTHONPATH_VALUE}"
    "CKL_OPT=${CKL_OPT}"
    "${Python3_EXECUTABLE}" "${INPUT_FILE}"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Python callable-alternative validation failed:\n${error}")
endif()
foreach(expected
    "alternative = \"direct\""
    "alternative = \"permuted\""
    "implementation = \"direct_impl\""
    "implementation = \"permuted_impl\""
    "gpu.func @device_boundary"
    "gpu.func @nested_device"
    "alternative = \"composite\""
    "alternative = \"leaf\""
    "arith.constant 41 : index"
    "arith.constant 42 : index")
  string(FIND "${output}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "Python callable output is missing ${expected}:\n${output}")
  endif()
endforeach()
