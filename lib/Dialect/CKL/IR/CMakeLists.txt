add_mlir_dialect_library(CKLDialect
  CKLDialect.cpp
  CKLInterfaces.cpp
  CKLOps.cpp
  CKLTypes.cpp

  DEPENDS
  MLIRCKLInterfacesIncGen
  MLIRCKLOpsIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
)
