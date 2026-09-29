#include "ckl/Analysis/NvidiaHostEmitter.h"
#include "ckl/Dialect/CKL/IR/CKLDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

namespace {
llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional, llvm::cl::desc("<input MLIR>"),
                                        llvm::cl::init("-"));
llvm::cl::opt<std::string> outputFilename("o", llvm::cl::desc("Output C++ file"),
                                         llvm::cl::value_desc("filename"), llvm::cl::init("-"));
} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "CKL NVIDIA host-plan generator\n");

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<ckl::CKLDialect>();
  MLIRContext context(registry);
  llvm::SourceMgr sourceManager;
  std::string errorMessage;
  auto input = mlir::openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << '\n';
    return 1;
  }
  sourceManager.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceManager, &context);
  if (!module)
    return 1;

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << '\n';
    return 1;
  }
  if (failed(ckl::emitNvidiaHostBuilders(*module, output->os())))
    return 1;
  output->keep();
  return 0;
}
