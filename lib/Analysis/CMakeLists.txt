add_mlir_library(CKLAnalysis
  EffectAnalysis.cpp
  GraphAlgorithms.cpp
  GraphAnalysis.cpp
  NvidiaHostEmitter.cpp

  LINK_LIBS PUBLIC
  CKLCore
  CKLDialect
  CKLMemoryPlanner
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSideEffectInterfaces
)
