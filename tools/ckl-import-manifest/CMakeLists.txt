add_llvm_executable(ckl-import-manifest ckl-import-manifest.cpp)
llvm_update_compile_flags(ckl-import-manifest)
target_link_libraries(ckl-import-manifest PRIVATE
  CKLExecDialect
  MLIRAsmParser
  MLIRIR
  MLIRSupport
  LLVMSupport
)
mlir_check_all_link_libraries(ckl-import-manifest)
