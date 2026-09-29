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
  set(generated_mlir "${CMAKE_CURRENT_BINARY_DIR}/generated-kernel.mlir")
  set(generated_cubin "${CMAKE_CURRENT_BINARY_DIR}/generated-kernel.cubin")
  add_custom_command(
    OUTPUT "${generated_mlir}"
    COMMAND $<TARGET_FILE:ckl-opt> "${CMAKE_CURRENT_SOURCE_DIR}/generated-kernel.mlir"
      "--gpu-lower-to-nvvm-pipeline=cubin-chip=sm_${CKL_RUNTIME_TEST_ARCH} cubin-features=+ptx87 cubin-format=bin"
      -o "${generated_mlir}"
    DEPENDS ckl-opt generated-kernel.mlir
    VERBATIM
  )
  add_custom_command(
    OUTPUT "${generated_cubin}"
    COMMAND ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${PROJECT_SOURCE_DIR}/python:$ENV{PYTHONPATH}"
      ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/extract_gpu_object.py"
      "${generated_mlir}" "${generated_cubin}"
    DEPENDS "${generated_mlir}" extract_gpu_object.py
    VERBATIM
  )
  add_custom_target(ckl-generated-test-cubin DEPENDS "${generated_cubin}")

  add_executable(ckl-generated-cubin-validation GeneratedCubinValidation.cpp)
  add_dependencies(ckl-generated-cubin-validation ckl-generated-test-cubin)
  target_link_libraries(ckl-generated-cubin-validation PRIVATE CKLNvidiaRuntime)
  target_compile_features(ckl-generated-cubin-validation PRIVATE cxx_std_17)
  target_compile_options(ckl-generated-cubin-validation PRIVATE -Wall -Wextra -Wpedantic)

  add_test(
    NAME ckl-generated-cubin-runtime
    COMMAND ckl-generated-cubin-validation "${generated_cubin}"
  )

  set(host_plan_mlir "${CMAKE_CURRENT_BINARY_DIR}/host-plan.graph.mlir")
  set(host_plan_cpp "${CMAKE_CURRENT_BINARY_DIR}/generated-host-plan.inc")
  add_custom_command(
    OUTPUT "${host_plan_mlir}"
    COMMAND $<TARGET_FILE:ckl-opt> "${CMAKE_CURRENT_SOURCE_DIR}/host-plan.mlir"
      --ckl-build-graph -o "${host_plan_mlir}"
    DEPENDS ckl-opt host-plan.mlir
    VERBATIM
  )
  add_custom_command(
    OUTPUT "${host_plan_cpp}"
    COMMAND $<TARGET_FILE:ckl-hostgen> "${host_plan_mlir}" -o "${host_plan_cpp}"
    DEPENDS ckl-hostgen "${host_plan_mlir}"
    VERBATIM
  )
  add_custom_target(ckl-generated-host-plan DEPENDS "${host_plan_cpp}")

  add_executable(ckl-generated-host-plan-validation GeneratedHostPlanValidation.cpp)
  add_dependencies(ckl-generated-host-plan-validation
    ckl-generated-host-plan
    ckl-runtime-test-cubin
  )
  target_include_directories(ckl-generated-host-plan-validation PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"
  )
  target_link_libraries(ckl-generated-host-plan-validation PRIVATE CKLNvidiaRuntime)
  target_compile_features(ckl-generated-host-plan-validation PRIVATE cxx_std_17)
  target_compile_options(ckl-generated-host-plan-validation PRIVATE -Wall -Wextra -Wpedantic)

  add_test(
    NAME ckl-generated-host-plan-runtime
    COMMAND ckl-generated-host-plan-validation "${runtime_cubin}"
  )
endif()
