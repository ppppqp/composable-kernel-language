add_executable(ckl-milestone0 milestone0.cu)
target_link_libraries(ckl-milestone0 PRIVATE CUDA::cudart)
target_compile_features(ckl-milestone0 PRIVATE cxx_std_17)
set_target_properties(ckl-milestone0 PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF
)
target_compile_options(ckl-milestone0 PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
