add_mlir_library(CKLTransforms
  ExecutablePlan.cpp
  CudaKernels.cpp
  PointwiseFusion.cpp

  LINK_LIBS PUBLIC
  CKLAnalysis
  CKLDialect
  CKLExecDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRIR
  MLIRPass
)
