if(CKL_ENABLE_CUDA_RUNTIME)
set(CKL_RUNTIME_TEST_ARCH "120" CACHE STRING "SM architecture for the runtime validation CUBIN")
set(CKL_CUDA_HOST_COMPILER "" CACHE FILEPATH "Host compiler used to build the validation CUBIN")
set(runtime_cubin "${CMAKE_CURRENT_BINARY_DIR}/runtime-kernels.cubin")
set(runtime_nvcc_host_args)
if(CKL_CUDA_HOST_COMPILER)
  list(APPEND runtime_nvcc_host_args "-ccbin=${CKL_CUDA_HOST_COMPILER}")
endif()

add_custom_command(
  OUTPUT "${runtime_cubin}"
  COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --cubin "-arch=sm_${CKL_RUNTIME_TEST_ARCH}"
    ${runtime_nvcc_host_args} "${CMAKE_CURRENT_SOURCE_DIR}/kernels.cu" -o "${runtime_cubin}"
  DEPENDS kernels.cu
  VERBATIM
)
add_custom_target(ckl-runtime-test-cubin DEPENDS "${runtime_cubin}")

add_executable(ckl-nvidia-runtime-validation NvidiaRuntimeValidation.cpp)
add_dependencies(ckl-nvidia-runtime-validation ckl-runtime-test-cubin)
target_link_libraries(ckl-nvidia-runtime-validation PRIVATE CKLNvidiaRuntime CUDA::cuda_driver)
target_compile_features(ckl-nvidia-runtime-validation PRIVATE cxx_std_17)
target_compile_options(ckl-nvidia-runtime-validation PRIVATE -Wall -Wextra -Wpedantic)

add_test(
  NAME ckl-nvidia-runtime
  COMMAND ckl-nvidia-runtime-validation "${runtime_cubin}"
    --elements 512 --cycles 3 --warmup 1 --repeats 1
)

add_executable(ckl-milestone4-validation Milestone4Validation.cpp)
add_dependencies(ckl-milestone4-validation ckl-runtime-test-cubin)
target_link_libraries(ckl-milestone4-validation PRIVATE
  CKLNvidiaRuntime
  CKLMemoryPlanner
)
target_compile_features(ckl-milestone4-validation PRIVATE cxx_std_17)
target_compile_options(ckl-milestone4-validation PRIVATE -Wall -Wextra -Wpedantic)

add_test(
  NAME ckl-milestone4
  COMMAND ckl-milestone4-validation "${runtime_cubin}"
    --elements 512 --cycles 8 --warmup 1 --repeats 1
)

if(CKL_ENABLE_MLIR)
  add_test(NAME ckl-cuda-frontend-runtime
    COMMAND ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${PROJECT_SOURCE_DIR}/python:${CKL_MLIR_PYTHON_PATH}:$ENV{PYTHONPATH}"
      "CKL_NATIVE_LIBRARY=$<TARGET_FILE:CKLCompiler>"
      ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_frontend_runtime.py"
      --toolkit "${CUDAToolkit_BIN_DIR}/.." --chip "sm_${CKL_RUNTIME_TEST_ARCH}")
  set_tests_properties(ckl-cuda-frontend-runtime PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME ckl-cuda-pointwise-fusion
    COMMAND ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${PROJECT_SOURCE_DIR}/python:${CKL_MLIR_PYTHON_PATH}:$ENV{PYTHONPATH}"
      "CKL_NATIVE_LIBRARY=$<TARGET_FILE:CKLCompiler>"
      ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_pointwise_fusion.py"
      --toolkit "${CUDAToolkit_BIN_DIR}/.." --chip "sm_${CKL_RUNTIME_TEST_ARCH}")
  set_tests_properties(ckl-cuda-pointwise-fusion PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME ckl-cuda-kernel-loops
    COMMAND ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${PROJECT_SOURCE_DIR}/python:${CKL_MLIR_PYTHON_PATH}:$ENV{PYTHONPATH}"
      "CKL_NATIVE_LIBRARY=$<TARGET_FILE:CKLCompiler>"
      ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_kernel_loops.py"
      --toolkit "${CUDAToolkit_BIN_DIR}/.." --chip "sm_${CKL_RUNTIME_TEST_ARCH}")

endif() # CKL_ENABLE_MLIR
endif() # CKL_ENABLE_CUDA_RUNTIME

if(CKL_ENABLE_HIP_RUNTIME)
  set(CKL_HIP_RUNTIME_TEST_ARCH "gfx942" CACHE STRING
    "GFX architecture for the runtime validation HSACO")
  if(NOT CKL_HIPCC_EXECUTABLE)
    unset(CKL_HIPCC_EXECUTABLE CACHE)
    find_program(CKL_HIPCC_EXECUTABLE hipcc
      HINTS /opt/rocm/bin /opt/rocm/core-10.0/bin
      REQUIRED
    )
  endif()
  get_filename_component(ckl_hip_bin_dir "${CKL_HIPCC_EXECUTABLE}" DIRECTORY)
  find_program(CKL_CLANG_OFFLOAD_BUNDLER_EXECUTABLE clang-offload-bundler
    HINTS "${ckl_hip_bin_dir}/../lib/llvm/bin" /opt/rocm/core-10.0/lib/llvm/bin
    REQUIRED
  )

  set(runtime_hsaco "${CMAKE_CURRENT_BINARY_DIR}/runtime-kernels.hsaco")
  set(runtime_hsaco_bundle "${CMAKE_CURRENT_BINARY_DIR}/runtime-kernels.bundle")
  add_custom_command(
    OUTPUT "${runtime_hsaco}"
    COMMAND "${CKL_HIPCC_EXECUTABLE}" --genco
      "--offload-arch=${CKL_HIP_RUNTIME_TEST_ARCH}"
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels.hip" -o "${runtime_hsaco_bundle}"
    COMMAND "${CKL_CLANG_OFFLOAD_BUNDLER_EXECUTABLE}" --unbundle --type=o
      "--targets=hipv4-amdgcn-amd-amdhsa--${CKL_HIP_RUNTIME_TEST_ARCH}"
      "--input=${runtime_hsaco_bundle}" "--output=${runtime_hsaco}"
    DEPENDS kernels.hip
    BYPRODUCTS "${runtime_hsaco_bundle}"
    VERBATIM
  )
  add_custom_target(ckl-runtime-test-hsaco DEPENDS "${runtime_hsaco}")

  add_executable(ckl-hip-runtime-validation HipRuntimeValidation.cpp)
  add_dependencies(ckl-hip-runtime-validation ckl-runtime-test-hsaco)
  target_link_libraries(ckl-hip-runtime-validation PRIVATE CKLHipRuntime)
  target_compile_features(ckl-hip-runtime-validation PRIVATE cxx_std_17)
  target_compile_options(ckl-hip-runtime-validation PRIVATE -Wall -Wextra -Wpedantic)

  add_test(
    NAME ckl-hip-runtime
    COMMAND ckl-hip-runtime-validation "${runtime_hsaco}"
  )
  set_tests_properties(ckl-hip-runtime PROPERTIES SKIP_RETURN_CODE 77)
endif()
