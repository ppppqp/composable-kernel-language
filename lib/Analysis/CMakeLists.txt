add_mlir_library(CKLAnalysis
  EffectAnalysis.cpp

  LINK_LIBS PUBLIC
  CKLDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
)
