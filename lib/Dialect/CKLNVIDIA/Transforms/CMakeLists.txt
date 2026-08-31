add_mlir_dialect_library(MLIRCKLNVIDIATransforms
  MaterializeMmaSync.cpp

  LINK_LIBS PUBLIC
  MLIRCKLDialect
  MLIRCKLNVIDIADialect
  MLIRIR
  MLIRPass
)

target_compile_options(MLIRCKLNVIDIATransforms PRIVATE -fexceptions)
target_compile_options(obj.MLIRCKLNVIDIATransforms PRIVATE -fexceptions)
