add_llvm_executable(ckl-hostgen ckl-hostgen.cpp)
llvm_update_compile_flags(ckl-hostgen)
target_link_libraries(ckl-hostgen PRIVATE
  CKLAnalysis
  CKLDialect
  CKLExecDialect
  MLIRIR
  MLIRParser
  MLIRRegisterAllDialects
  MLIRSupport
)
mlir_check_all_link_libraries(ckl-hostgen)
