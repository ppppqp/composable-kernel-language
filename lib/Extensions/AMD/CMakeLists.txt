add_library(CKLAMDExtension
  MfmaLayouts.cpp
)

target_link_libraries(CKLAMDExtension PUBLIC CKL::Core)
target_include_directories(CKLAMDExtension
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_compile_options(CKLAMDExtension PRIVATE -Wall -Wextra -Wpedantic)

add_library(CKL::AMD ALIAS CKLAMDExtension)
