add_library(CKLCompiler SHARED Compiler.cpp)
target_link_libraries(CKLCompiler PRIVATE
  CKLTransforms CKLAnalysis CKLDialect CKLExecDialect
  MLIRRegisterAllPasses MLIRRegisterAllDialects MLIRRegisterAllExtensions
  MLIRToLLVMIRTranslationRegistration MLIRExecutionEngine MLIRParser)
set_target_properties(CKLCompiler PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python/ckl")
if(CKL_ENABLE_CUDA_RUNTIME)
  target_compile_definitions(CKLCompiler PRIVATE CKL_HAS_CUDA_RUNTIME=1)
  target_link_libraries(CKLCompiler PRIVATE CKLNvidiaRuntime)
endif()
