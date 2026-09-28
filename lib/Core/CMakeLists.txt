add_mlir_library(CKLCore
  MemoryEffects.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
)
