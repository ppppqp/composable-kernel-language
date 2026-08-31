add_mlir_dialect_library(MLIRCKLNVIDIATransforms
  MaterializeMmaSync.cpp
  LowerMmaSync.cpp

  LINK_LIBS PUBLIC
  MLIRCKLDialect
  MLIRCKLNVIDIADialect
  MLIRIR
  MLIRPass
  MLIRNVGPUDialect
)

target_compile_options(MLIRCKLNVIDIATransforms PRIVATE -fexceptions)
target_compile_options(obj.MLIRCKLNVIDIATransforms PRIVATE -fexceptions)
