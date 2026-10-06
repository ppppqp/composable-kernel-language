if(CKL_ENABLE_CUDA_RUNTIME)
  add_library(CKLNvidiaRuntime
    NvidiaRuntime.cpp
  )
  target_include_directories(CKLNvidiaRuntime PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  )
  target_compile_features(CKLNvidiaRuntime PUBLIC cxx_std_17)
  target_compile_options(CKLNvidiaRuntime PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(CKLNvidiaRuntime PUBLIC CKLRuntimeCore CUDA::cuda_driver)
endif()

if(CKL_ENABLE_HIP_RUNTIME)
  add_library(CKLHipRuntime
    HipRuntime.cpp
  )
  target_include_directories(CKLHipRuntime PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    ${CKL_HIP_INCLUDE_DIR}
  )
  target_compile_features(CKLHipRuntime PUBLIC cxx_std_17)
  target_compile_definitions(CKLHipRuntime PUBLIC __HIP_PLATFORM_AMD__=1)
  target_compile_options(CKLHipRuntime PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(CKLHipRuntime PUBLIC CKLRuntimeCore ${CKL_HIP_LIBRARY})
endif()
