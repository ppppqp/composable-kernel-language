add_subdirectory(Core)

if(CKL_ENABLE_MLIR)
  add_subdirectory(Dialect)
  add_subdirectory(Analysis)
  add_subdirectory(Transforms)
endif()

if(CKL_ENABLE_CUDA_RUNTIME OR CKL_ENABLE_HIP_RUNTIME)
  add_subdirectory(Runtime)
endif()
