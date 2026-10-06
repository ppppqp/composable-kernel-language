set(CKL_HIP_BENCHMARK_ARCH "gfx1200" CACHE STRING
  "GFX architecture for the HIP runtime-overhead benchmark")

if(NOT CKL_HIPCC_EXECUTABLE)
  unset(CKL_HIPCC_EXECUTABLE CACHE)
  find_program(CKL_HIPCC_EXECUTABLE hipcc
    HINTS /opt/rocm/bin /opt/rocm/core-10.0/bin
    REQUIRED
  )
endif()
get_filename_component(ckl_hip_benchmark_bin_dir "${CKL_HIPCC_EXECUTABLE}" DIRECTORY)
find_program(CKL_HIP_BENCHMARK_BUNDLER clang-offload-bundler
  HINTS "${ckl_hip_benchmark_bin_dir}/../lib/llvm/bin" /opt/rocm/core-10.0/lib/llvm/bin
  REQUIRED
)

set(benchmark_hsaco "${CMAKE_CURRENT_BINARY_DIR}/overhead-kernel.hsaco")
set(benchmark_bundle "${CMAKE_CURRENT_BINARY_DIR}/overhead-kernel.bundle")
add_custom_command(
  OUTPUT "${benchmark_hsaco}"
  COMMAND "${CKL_HIPCC_EXECUTABLE}" --genco
    "--offload-arch=${CKL_HIP_BENCHMARK_ARCH}"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel.hip" -o "${benchmark_bundle}"
  COMMAND "${CKL_HIP_BENCHMARK_BUNDLER}" --unbundle --type=o
    "--targets=hipv4-amdgcn-amd-amdhsa--${CKL_HIP_BENCHMARK_ARCH}"
    "--input=${benchmark_bundle}" "--output=${benchmark_hsaco}"
  DEPENDS kernel.hip
  BYPRODUCTS "${benchmark_bundle}"
  VERBATIM
)
add_custom_target(ckl-hip-overhead-hsaco DEPENDS "${benchmark_hsaco}")

add_executable(ckl-hip-runtime-overhead overhead.cpp)
add_dependencies(ckl-hip-runtime-overhead ckl-hip-overhead-hsaco)
target_link_libraries(ckl-hip-runtime-overhead PRIVATE CKLHipRuntime)
target_compile_features(ckl-hip-runtime-overhead PRIVATE cxx_std_17)
target_compile_options(ckl-hip-runtime-overhead PRIVATE -Wall -Wextra -Wpedantic)

add_custom_target(run-ckl-hip-runtime-overhead
  COMMAND $<TARGET_FILE:ckl-hip-runtime-overhead> "${benchmark_hsaco}"
  DEPENDS ckl-hip-runtime-overhead
  USES_TERMINAL
)
