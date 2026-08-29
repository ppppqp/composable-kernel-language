add_library(CKLCore
  IndexMap.cpp
  Distribution.cpp
  StorageLayout.cpp
  ../Composition/Task.cpp
  ../Planning/ExchangeSchedule.cpp
)

if(CKL_ENABLE_MLIR)
  target_sources(CKLCore PRIVATE ../Proof/SymbolicProof.cpp)
  target_link_libraries(CKLCore PUBLIC MLIRPresburger)
  target_compile_definitions(CKLCore PUBLIC CKL_ENABLE_MLIR=1)
endif()

target_include_directories(CKLCore
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_compile_options(CKLCore PRIVATE -Wall -Wextra -Wpedantic)

add_library(CKL::Core ALIAS CKLCore)
