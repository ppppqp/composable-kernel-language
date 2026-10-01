add_mlir_library(CKLTransforms
  ExecutablePlan.cpp

  LINK_LIBS PUBLIC
  CKLAnalysis
  CKLDialect
  CKLExecDialect
  MLIRIR
  MLIRPass
)
