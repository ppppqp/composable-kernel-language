add_subdirectory(Core)
if(CKL_ENABLE_MLIR)
  add_subdirectory(Dialect)
endif()
