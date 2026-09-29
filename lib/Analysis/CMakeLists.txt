add_mlir_library(CKLAnalysis
  EffectAnalysis.cpp
  GraphAnalysis.cpp

  LINK_LIBS PUBLIC
  CKLCore
  CKLDialect
  CKLMemoryPlanner
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSideEffectInterfaces
)
