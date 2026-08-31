add_mlir_dialect_library(MLIRCKLTransforms
  PlanCompositions.cpp
  EnumerateAlternatives.cpp

  LINK_LIBS PUBLIC
  CKL::Core
  MLIRCKLDialect
  MLIRIR
  MLIRPass
)

target_compile_options(MLIRCKLTransforms PRIVATE -fexceptions)
target_compile_options(obj.MLIRCKLTransforms PRIVATE -fexceptions)
