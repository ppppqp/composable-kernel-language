add_library(CKLNvidiaRuntime
  NvidiaRuntime.cpp
)
target_include_directories(CKLNvidiaRuntime PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_compile_features(CKLNvidiaRuntime PUBLIC cxx_std_17)
target_compile_options(CKLNvidiaRuntime PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(CKLNvidiaRuntime PUBLIC CUDA::cuda_driver)
