if(CKL_ENABLE_MLIR)
  add_subdirectory(Core)
  add_subdirectory(Dialect)
  add_subdirectory(Analysis)
endif()

if(CKL_ENABLE_CUDA_RUNTIME)
  add_subdirectory(Runtime)
endif()
