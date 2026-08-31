add_llvm_executable(ckl-opt ckl-opt.cpp)
llvm_update_compile_flags(ckl-opt)
target_link_libraries(ckl-opt PRIVATE
  MLIRCKLDialect
  MLIRCKLTransforms
  MLIRCKLNVIDIADialect
  MLIRCKLNVIDIATransforms
  MLIRFuncDialect
  MLIROptLib
  MLIRRegisterAllPasses
  CKL::NVIDIA
)
mlir_check_all_link_libraries(ckl-opt)
