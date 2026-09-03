add_mlir_dialect_library(MLIRCKLTransforms
  PlanCompositions.cpp
  EnumerateAlternatives.cpp
  MaterializeSelectedAlternatives.cpp
  ResolveTaskAlternatives.cpp

  LINK_LIBS PUBLIC
  CKL::Core
  MLIRCKLDialect
  MLIRIR
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
)

target_compile_options(MLIRCKLTransforms PRIVATE -fexceptions)
target_compile_options(obj.MLIRCKLTransforms PRIVATE -fexceptions)
