execute_process(
  COMMAND "${CKL_OPT}" "${INPUT_FILE}"
          --ckl-nvidia-materialize-mma-sync --ckl-nvidia-lower-mma-sync
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "ckl-opt failed:\n${error}")
endif()
if(NOT output MATCHES "nvgpu.mma.sync")
  message(FATAL_ERROR "NVIDIA MMA did not lower to NVGPU:\n${output}")
endif()
if(output MATCHES "ckl_nvidia.mma_sync")
  message(FATAL_ERROR "selected extension MMA survived NVGPU lowering:\n${output}")
endif()
foreach(role lhs rhs acc result)
  if(NOT output MATCHES "role = \"${role}\"")
    message(FATAL_ERROR "missing explicit ${role} fragment boundary:\n${output}")
  endif()
endforeach()
if(NOT output MATCHES "vector<4x2xf16>" OR
   NOT output MATCHES "vector<2x2xf16>" OR
   NOT output MATCHES "vector<2x2xf32>")
  message(FATAL_ERROR "unexpected fixed MMA register fragment types:\n${output}")
endif()
string(FIND "${output}" "mmaShape = [16, 8, 16]" mma_shape_position)
if(mma_shape_position EQUAL -1)
  message(FATAL_ERROR "unexpected NVGPU MMA shape:\n${output}")
endif()

set(lowered_file "${CMAKE_CURRENT_BINARY_DIR}/nvidia_mma_lowered.mlir")
file(WRITE "${lowered_file}" "${output}")
execute_process(
  COMMAND "${CKL_OPT}" "${lowered_file}"
  RESULT_VARIABLE roundtrip_result ERROR_VARIABLE roundtrip_error)
if(NOT roundtrip_result EQUAL 0)
  message(FATAL_ERROR "NVGPU handoff IR did not round-trip:\n${roundtrip_error}")
endif()
