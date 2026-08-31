add_mlir_dialect_library(MLIRCKLNVIDIADialect
  CKLNVIDIADialect.cpp
  CKLNVIDIAOps.cpp

  DEPENDS
  MLIRCKLNVIDIAOpsIncGen

  LINK_LIBS PUBLIC
  CKL::Core
  MLIRCKLDialect
  MLIRIR
)

target_compile_options(MLIRCKLNVIDIADialect PRIVATE -fexceptions)
target_compile_options(obj.MLIRCKLNVIDIADialect PRIVATE -fexceptions)
