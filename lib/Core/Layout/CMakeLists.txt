add_library(CKLCore
  IndexMap.cpp
  Distribution.cpp
)

target_include_directories(CKLCore
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_compile_options(CKLCore PRIVATE -Wall -Wextra -Wpedantic)

add_library(CKL::Core ALIAS CKLCore)

