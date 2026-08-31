add_library(CKLNVIDIAExtension
  MmaSync.cpp
)

target_link_libraries(CKLNVIDIAExtension PUBLIC CKL::Core)
target_include_directories(CKLNVIDIAExtension
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_compile_options(CKLNVIDIAExtension PRIVATE -Wall -Wextra -Wpedantic)

add_library(CKL::NVIDIA ALIAS CKLNVIDIAExtension)
