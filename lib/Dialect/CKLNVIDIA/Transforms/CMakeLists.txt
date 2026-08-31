add_mlir_dialect_library(MLIRCKLNVIDIATransforms
  MaterializeMmaSync.cpp
  LowerMmaSync.cpp
  LowerFragmentIO.cpp

  LINK_LIBS PUBLIC
  MLIRCKLDialect
  MLIRCKLNVIDIADialect
  MLIRIR
  MLIRPass
  MLIRNVGPUDialect
  MLIRArithDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRVectorDialect
  CKL::NVIDIA
)

target_compile_options(MLIRCKLNVIDIATransforms PRIVATE -fexceptions)
target_compile_options(obj.MLIRCKLNVIDIATransforms PRIVATE -fexceptions)
