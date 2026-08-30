add_mlir_dialect(CKLOps ckl)

set(LLVM_TARGET_DEFINITIONS CKLAttrs.td)
mlir_tablegen(CKLAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=ckl)
mlir_tablegen(CKLAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ckl)
add_public_tablegen_target(CKLAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS CKLTypes.td)
mlir_tablegen(CKLTypes.h.inc -gen-typedef-decls -typedefs-dialect=ckl)
mlir_tablegen(CKLTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ckl)
add_public_tablegen_target(CKLTypesIncGen)
