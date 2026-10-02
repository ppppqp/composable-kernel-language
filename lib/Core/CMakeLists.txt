add_library(CKLMemoryPlanner
  MemoryPlanner.cpp
)
target_include_directories(CKLMemoryPlanner PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_compile_features(CKLMemoryPlanner PUBLIC cxx_std_17)
target_compile_options(CKLMemoryPlanner PRIVATE -Wall -Wextra -Wpedantic)

add_library(CKLRuntimeCore
  ExecutionPlan.cpp
)
target_include_directories(CKLRuntimeCore PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_compile_features(CKLRuntimeCore PUBLIC cxx_std_17)
target_compile_options(CKLRuntimeCore PRIVATE -Wall -Wextra -Wpedantic)

if(CKL_ENABLE_MLIR)
  add_mlir_library(CKLCore
    MemoryEffects.cpp

    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSideEffectInterfaces
    MLIRViewLikeInterface
  )
endif()
