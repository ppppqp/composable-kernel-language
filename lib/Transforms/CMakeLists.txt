add_mlir_library(CKLTransforms
  ExecutablePlan.cpp
  CudaKernels.cpp

  LINK_LIBS PUBLIC
  CKLAnalysis
  CKLDialect
  CKLExecDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRIR
  MLIRPass
)
