add_mlir_dialect_library(CKLExecDialect
  CKLExecDialect.cpp
  CKLExecOps.cpp

  DEPENDS
  MLIRCKLExecOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
