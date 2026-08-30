add_mlir_dialect_library(MLIRCKLDialect
  CKLAttrs.cpp
  CKLDialect.cpp
  CKLOps.cpp
  CKLTypes.cpp

  DEPENDS
  CKLAttrsIncGen
  MLIRCKLOpsIncGen
  CKLTypesIncGen

  LINK_LIBS PUBLIC
  CKL::Core
  MLIRIR
)

target_compile_options(MLIRCKLDialect PRIVATE -fexceptions)
target_compile_options(obj.MLIRCKLDialect PRIVATE -fexceptions)
