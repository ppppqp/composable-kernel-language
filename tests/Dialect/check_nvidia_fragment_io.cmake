execute_process(
  COMMAND "${CKL_OPT}" "${INPUT_FILE}" --ckl-nvidia-materialize-mma-sync
          --ckl-nvidia-lower-mma-sync --ckl-nvidia-lower-fragment-io
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "fixed NVIDIA pipeline failed:\n${error}")
endif()
if(NOT output MATCHES "nvgpu.mma.sync" OR NOT output MATCHES "gpu.thread_id")
  message(FATAL_ERROR "missing NVGPU instruction or lane addressing:\n${output}")
endif()
if(output MATCHES "ckl.load_tile" OR output MATCHES "ckl.store_tile" OR
   output MATCHES "ckl_nvidia.pack_fragment" OR output MATCHES "ckl_nvidia.unpack_fragment")
  message(FATAL_ERROR "fragment memory boundaries were not fully lowered:\n${output}")
endif()
string(REGEX MATCHALL "memref.load" loads "${output}")
list(LENGTH loads load_count)
if(NOT load_count EQUAL 16)
  message(FATAL_ERROR "expected 16 per-lane fragment loads, got ${load_count}:\n${output}")
endif()
string(REGEX MATCHALL "memref.store" stores "${output}")
list(LENGTH stores store_count)
if(NOT store_count EQUAL 4)
  message(FATAL_ERROR "expected four per-lane result stores, got ${store_count}:\n${output}")
endif()
