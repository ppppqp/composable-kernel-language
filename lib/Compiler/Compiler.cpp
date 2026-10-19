#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/array.h>
#include <array>
#include <cstring>

// In-process CKL compiler and host JIT, exposed as typed nanobind objects.
#include "ckl/Analysis/EffectAnalysis.h"
#include "ckl/Dialect/CKL/IR/CKLDialect.h"
#include "ckl/Dialect/Exec/IR/CKLExecDialect.h"
#include "ckl/Dialect/Exec/IR/CKLExecOps.h"
#include "ckl/Transforms/CudaKernels.h"
#include "ckl/Transforms/ExecutablePlan.h"
#include "ckl/Transforms/PointwiseFusion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CKL_HAS_CUDA_RUNTIME
#include "ckl/Runtime/NvidiaRuntime.h"
#include <cuda.h>
#include <cudaProfiler.h>
#endif

namespace nb = nanobind;
using namespace mlir;
namespace {
void initialize() {
  static std::once_flag once;
  std::call_once(once, [] {
    registerAllPasses();
    ckl::registerCKLPasses();
    ckl::registerCKLExecutablePlanPasses();
    ckl::registerCKLCudaKernelPasses();
    ckl::registerCKLPointwiseFusionPass();
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  });
}

std::string print(Operation *op) {
  std::string result;
  llvm::raw_string_ostream out(result);
  op->print(out);
  return result;
}

void pipeline(ModuleOp module, StringRef text, StringRef toolkit = {}) {
  PassManager pm(module.getContext());
  std::string error;
  llvm::raw_string_ostream out(error);
  if (failed(parsePassPipeline(text, pm, out)))
    throw std::runtime_error(error);
  // Set the toolkit directly on the upstream serializer, without mutating
  // process-global CUDA_HOME/PATH while Python or other compilers are running.
  if (!toolkit.empty())
    for (Pass &pass : pm.getPasses())
      if (pass.getArgument() == "gpu-module-to-binary") {
        std::string options = "format=bin toolkit=\"" + toolkit.str() + "\"";
        if (failed(pass.initializeOptions(options, [&](const llvm::Twine &message) {
              out << message;
              return failure();
            })))
          throw std::runtime_error("invalid CUDA toolkit path");
      }
  if (failed(pm.run(module)))
    throw std::runtime_error("CKL compilation failed");
}

struct Program {
  // Context must outlive its IR and ExecutionEngine, as in FlyDSL.
  MLIRContext context;
  OwningOpRef<ModuleOp> source, executable, device;
  std::string preparedIR, executableIR, deviceIR, hostIR, diagnostics;
  std::unique_ptr<ExecutionEngine> engine;
  using Entry = void (*)(void *, void **);
  Entry entry = nullptr;
  unsigned argumentCount = 0;
  std::vector<uint64_t> bufferBytes;
  std::vector<std::string> scalarTypes;
  std::shared_ptr<Program> aliasFallback;
#ifdef CKL_HAS_CUDA_RUNTIME
  std::mutex moduleMutex;
  std::map<uint64_t, ckl::runtime::NvidiaModule> loadedModules;
#endif
};

#ifdef CKL_HAS_CUDA_RUNTIME
using namespace ckl::runtime;
std::atomic<uint64_t> nextDeviceId{0};
struct Device {
  NvidiaRuntime runtime;
  const uint64_t id = nextDeviceId++;
};
struct Buffer {
  std::shared_ptr<Device> device;
  NvidiaBuffer buffer;
};
struct Invocation {
  std::shared_ptr<Program> program;
  std::shared_ptr<Device> device;
  NvidiaStream stream;
  NvidiaModule module;
  std::vector<NvidiaBuffer> heaps;
  std::vector<std::shared_ptr<Buffer>> buffers;
  std::vector<uint64_t> resources;
  NvidiaPlan plan;
  KernelLaunch pending;
  std::vector<std::size_t> dependencies;
  std::map<uint64_t, std::size_t> nodes;
  NvidiaGraphExecutable graph;
  std::map<std::pair<int, uint64_t>, NvidiaGraphExecutable> executionGraphs;
  std::mutex runMutex;
  std::string error;
  Invocation(std::shared_ptr<Program> program, std::shared_ptr<Device> device)
      : program(std::move(program)), device(device), stream(device->runtime.createStream()) {}
};

// Exceptions never unwind through JIT frames. Each callback records failure and
// subsequent callbacks become no-ops; binding reports the error after entry returns.
template <typename F> void callback(void *opaque, F action) {
  auto &frame = *static_cast<Invocation *>(opaque);
  if (!frame.error.empty())
    return;
  try {
    action(frame);
  } catch (const std::exception &error) {
    frame.error = error.what();
  }
}
void begin(void *frame, const void *binary, uint64_t bytes) {
  callback(frame, [&](Invocation &f) {
    std::lock_guard<std::mutex> lock(f.program->moduleMutex);
    auto &module = f.program->loadedModules[f.device->id];
    if (!module)
      module = f.device->runtime.loadCubin(binary, bytes);
    f.module = module;
  });
}
void heap(void *frame, uint64_t id, uint64_t bytes) {
  callback(frame, [&](Invocation &f) {
    if (f.heaps.size() <= id)
      f.heaps.resize(id + 1);
    f.heaps[id] = f.device->runtime.allocate(bytes);
  });
}
void resource(void *frame, uint64_t id, uint64_t heapId, uint64_t offset, void *external) {
  callback(frame, [&](Invocation &f) {
    if (f.resources.size() <= id)
      f.resources.resize(id + 1);
    f.resources[id] =
        external ? *static_cast<uint64_t *>(external) : f.heaps.at(heapId).address() + offset;
  });
}
void kernel(void *frame, const char *name, uint64_t gx, uint64_t gy, uint64_t gz, uint64_t bx,
            uint64_t by, uint64_t bz, uint64_t shared) {
  callback(frame, [&](Invocation &f) {
    f.pending = {f.module.function(name),
                 {unsigned(gx), unsigned(gy), unsigned(gz)},
                 {unsigned(bx), unsigned(by), unsigned(bz)},
                 unsigned(shared),
                 {}};
    f.dependencies.clear();
  });
}
void argumentResource(void *frame, uint64_t id) {
  callback(frame, [&](Invocation &f) { f.pending.arguments.add(f.resources.at(id)); });
}
void argumentValue(void *frame, const void *value, uint64_t bytes) {
  callback(frame, [&](Invocation &f) { f.pending.arguments.addBytes(value, bytes); });
}
void dependency(void *frame, uint64_t id) {
  callback(frame, [&](Invocation &f) { f.dependencies.push_back(f.nodes.at(id)); });
}
void commit(void *frame, uint64_t id) {
  callback(frame, [&](Invocation &f) {
    f.nodes[id] = f.plan.addKernel(std::move(f.pending), std::move(f.dependencies));
  });
}

void validateExecution(Invocation &f, int mode, uint64_t batch) {
  if (mode < 0 || mode > 2 || !batch || f.plan.empty() || batch > 1000000 / f.plan.nodes().size())
    throw std::invalid_argument("invalid execution mode or batch exceeds one million kernel nodes");
}

NvidiaGraphExecutable &executionGraph(Invocation &f, int mode, uint64_t batch) {
  auto &graph = f.executionGraphs[{mode, batch}];
  if (!graph) {
    NvidiaPlan plan = batch == 1 ? f.plan : f.plan.repeat(batch);
    graph =
        mode == 2 ? f.device->runtime.capture(plan, f.stream) : f.device->runtime.instantiate(plan);
  }
  return graph;
}

void execute(Invocation &f, int mode, uint64_t batch) {
  if (mode)
    executionGraph(f, mode, batch).launch(f.stream);
  else
    for (uint64_t i = 0; i < batch; ++i)
      f.device->runtime.launchOrdinary(f.plan, f.stream);
}
#endif

// Emit host LLVM directly from the verified executable plan.
std::unique_ptr<llvm::Module> buildHost(Program &program, ckl::exec::PlanOp plan, StringRef binary,
                                        llvm::LLVMContext &context) {
  auto module = std::make_unique<llvm::Module>("ckl.program", context);
  llvm::IRBuilder<> b(context);
  auto *ptr = b.getPtrTy();
  auto *i64 = b.getInt64Ty();
  auto *function =
      llvm::Function::Create(llvm::FunctionType::get(b.getVoidTy(), {ptr, ptr}, false),
                             llvm::GlobalValue::ExternalLinkage, "ckl_entry", module.get());
  b.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", function));
  auto *frame = function->getArg(0);
  auto *arguments = function->getArg(1);
  auto number = [&](uint64_t value) { return b.getInt64(value); };
  auto call = [&](StringRef name, llvm::ArrayRef<llvm::Value *> values) {
    // emit a call to a host runtime function with the given name and arguments
    llvm::SmallVector<llvm::Type *> types;
    for (auto *value : values)
      types.push_back(value->getType());
    auto callee =
        module->getOrInsertFunction(name, llvm::FunctionType::get(b.getVoidTy(), types, false));
    b.CreateCall(callee, values);
  };
  auto arg = [&](StringRef name) -> llvm::Value * {
    unsigned index;
    if (!name.consume_front("arg") || name.getAsInteger(10, index) ||
        index >= program.argumentCount)
      throw std::runtime_error("host JIT requires original program argument bindings");
    return b.CreateLoad(ptr, b.CreateConstGEP1_64(ptr, arguments, index));
  };
  call("ckl_begin",
       {frame, b.CreateGlobalString(binary, "ckl.binary", 0, module.get()), number(binary.size())});
  std::map<std::string, uint64_t> resources;
  for (auto heap : plan.getBody().front().getOps<ckl::exec::HeapOp>())
    call("ckl_heap", {frame, number(heap.getId()), number(heap.getBytes())});
  for (auto resource : plan.getBody().front().getOps<ckl::exec::ResourceOp>()) {
    uint64_t id = resources.size();
    resources.emplace(resource.getName().str(), id);
    bool external = resource.getKind() == "external";
    call("ckl_resource",
         {frame, number(id), number(external ? 0 : resource.getHeapAttr().getInt()),
          number(external ? 0 : resource.getOffsetAttr().getInt()),
          external ? arg(resource.getName()) : llvm::ConstantPointerNull::get(ptr)});
  }
  for (auto kernel : plan.getBody().front().getOps<ckl::exec::KernelOp>()) {
    if (kernel.getAbi() != "cuda.direct" || kernel.getDevice() != "cuda:0")
      throw std::runtime_error("host JIT requires cuda.direct on logical device 0");
    auto grid = kernel.getGrid(), block = kernel.getBlock();
    call("ckl_kernel",
         {frame, b.CreateGlobalString(kernel.getKernel().getLeafReference().getValue()),
          number(grid[0]), number(grid[1]), number(grid[2]), number(block[0]), number(block[1]),
          number(block[2]), number(kernel.getSharedMemory())});
    SmallVector<DictionaryAttr> slots(kernel.getArguments().getAsRange<DictionaryAttr>());
    llvm::sort(slots, [](DictionaryAttr a, DictionaryAttr b) {
      return a.getAs<IntegerAttr>("slot").getInt() < b.getAs<IntegerAttr>("slot").getInt();
    });
    for (auto slot : slots) {
      StringRef kind = slot.getAs<StringAttr>("kind").getValue();
      if (kind == "resource") {
        call("ckl_argument_resource",
             {frame, number(resources.at(slot.getAs<StringAttr>("resource").getValue().str()))});
        continue;
      }
      uint64_t bytes = 0;
      llvm::Value *value = nullptr;
      if (kind == "bytes") {
        auto data = slot.getAs<DenseI8ArrayAttr>("value").asArrayRef();
        bytes = data.size();
        value = b.CreateGlobalString(StringRef(reinterpret_cast<const char *>(data.data()), bytes));
      } else {
        auto type = slot.getAs<TypeAttr>("type").getValue();
        bytes = type.getIntOrFloatBitWidth() / 8;
        if (kind == "scalar") {
          value = arg(slot.getAs<StringAttr>("name").getValue());
        } else if (kind == "constant") {
          uint64_t bits;
          if (auto integer = dyn_cast<IntegerAttr>(slot.get("value")))
            bits = integer.getValue().getZExtValue();
          else
            bits = cast<FloatAttr>(slot.get("value")).getValue().bitcastToAPInt().getZExtValue();
          std::string data(bytes, '\0');
          // NVPTX and supported host platforms use little-endian scalar ABI.
          for (uint64_t i = 0; i < bytes; ++i)
            data[i] = (bits >> (8 * i)) & 255;
          value = b.CreateGlobalString(data);
        } else {
          throw std::runtime_error("unsupported host argument binding");
        }
      }
      call("ckl_argument_value", {frame, value, number(bytes)});
    }
    for (int64_t predecessor : kernel.getDependencies())
      call("ckl_dependency", {frame, number(predecessor)});
    call("ckl_commit", {frame, number(kernel.getId())});
  }
  b.CreateRetVoid();
  if (llvm::verifyModule(*module, &llvm::errs()))
    throw std::runtime_error("invalid generated host LLVM module");
  llvm::raw_string_ostream out(program.hostIR);
  module->print(out, nullptr);
  return module;
}

std::shared_ptr<Program> compile(const char *source, const char *chip, const char *features,
                                 const char *toolkit, bool fusion, bool reuseStorage = true) {
  initialize();
  auto program = std::make_shared<Program>();
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registerAllGPUToLLVMIRTranslations(registry);
  registry.insert<ckl::CKLDialect, ckl::exec::CKLExecDialect>();
  program->context.appendDialectRegistry(registry);
  ScopedDiagnosticHandler handler(&program->context, [&](Diagnostic &diagnostic) {
    llvm::raw_string_ostream out(program->diagnostics);
    diagnostic.print(out);
    out << '\n';
    return success();
  });
  try {
    program->source = parseSourceString<ModuleOp>(source, &program->context);
    if (!program->source)
      throw std::runtime_error("invalid CKL source module");
    if ((*program->source)->hasAttr("ckl.fusion.requires_disjoint"))
      throw std::runtime_error("compile original source, not pre-fused guarded IR");
    if (fusion)
      pipeline(*program->source, "ckl-fuse-pointwise");
    bool needsFallback = (*program->source)->hasAttr("ckl.fusion.requires_disjoint");
    pipeline(*program->source,
             "ckl-prepare-cuda{chip=" + std::string(chip) + " features=" + features + "}");
    program->preparedIR = print(*program->source);
    program->executable = cast<ModuleOp>((*program->source)->clone());
    pipeline(*program->executable, std::string("ckl-build-graph{reuse-storage=") +
                                       (reuseStorage ? "true" : "false") +
                                       "},ckl-lower-graph-to-exec");
    program->executableIR = print(*program->executable);
    SmallVector<ckl::exec::PlanOp> plans(program->executable->getOps<ckl::exec::PlanOp>());
    if (plans.size() != 1)
      throw std::runtime_error("compilation requires exactly one host program");
    auto function =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(plans[0], plans[0].getSource());
    program->argumentCount = function.getNumArguments();
    for (Type type : function.getArgumentTypes()) {
      uint64_t bytes = 0;
      if (auto buffer = dyn_cast<MemRefType>(type))
        bytes = buffer.getNumElements() * (buffer.getElementTypeBitWidth() / 8);
      program->bufferBytes.push_back(bytes);
      std::string scalarType;
      if (!bytes) {
        llvm::raw_string_ostream out(scalarType);
        type.print(out);
      }
      program->scalarTypes.push_back(std::move(scalarType));
    }
    program->device = cast<ModuleOp>((*program->source)->clone());
    std::string devicePipeline =
        "ckl-lower-kernels-to-gpu,gpu-lower-to-nvvm-pipeline{cubin-chip=" + std::string(chip) +
        " cubin-features=" + features +
        " cubin-format=bin index-bitwidth=64 opt-level=3 kernel-bare-ptr-calling-convention=true}";
    pipeline(*program->device, devicePipeline, toolkit);
    program->deviceIR = print(*program->device);
    SmallVector<gpu::ObjectAttr> objects;
    program->device->walk([&](gpu::BinaryOp binary) {
      for (auto object : binary.getObjects())
        objects.push_back(cast<gpu::ObjectAttr>(object));
    });
    if (objects.size() != 1 || objects[0].getFormat() != gpu::CompilationTarget::Binary ||
        !isa<NVVM::NVVMTargetAttr>(objects[0].getTarget()) ||
        !objects[0].getObject().getValue().starts_with(StringRef("\x7f"
                                                                 "ELF",
                                                                 4)))
      throw std::runtime_error("expected exactly one NVIDIA ELF binary object");
    auto builder = [&](Operation *, llvm::LLVMContext &context) {
      return buildHost(*program, plans[0], objects[0].getObject().getValue(), context);
    };
    ExecutionEngineOptions options;
    options.llvmModuleBuilder = builder;
    auto engine = ExecutionEngine::create(*program->executable, options);
    if (!engine)
      throw std::runtime_error(llvm::toString(engine.takeError()));
    program->engine = std::move(*engine);
#ifdef CKL_HAS_CUDA_RUNTIME
    program->engine->registerSymbols([&](llvm::orc::MangleAndInterner mangle) {
      llvm::orc::SymbolMap symbols;
      auto add = [&](const char *name, auto pointer) {
        symbols[mangle(name)] = {llvm::orc::ExecutorAddr::fromPtr(pointer),
                                 llvm::JITSymbolFlags::Exported};
      };
      add("ckl_begin", &begin);
      add("ckl_heap", &heap);
      add("ckl_resource", &resource);
      add("ckl_kernel", &kernel);
      add("ckl_argument_resource", &argumentResource);
      add("ckl_argument_value", &argumentValue);
      add("ckl_dependency", &dependency);
      add("ckl_commit", &commit);
      return symbols;
    });
    auto entry = program->engine->lookup("ckl_entry");
    if (!entry)
      throw std::runtime_error(llvm::toString(entry.takeError()));
    program->entry = reinterpret_cast<Program::Entry>(*entry);
#endif
    if (needsFallback)
      program->aliasFallback = compile(source, chip, features, toolkit, false, reuseStorage);
    return program;
  } catch (const std::exception &error) {
    throw std::runtime_error(std::string(error.what()) + "\n" + program->diagnostics);
  }
}
#ifdef CKL_HAS_CUDA_RUNTIME
std::shared_ptr<Invocation> bindProgram(std::shared_ptr<Program> p, std::shared_ptr<Device> d,
                                      nb::sequence arguments) {
  const auto count = nb::len(arguments);
  if (count != p->argumentCount)
    throw std::invalid_argument("incorrect program argument count");
  std::vector<uint64_t> addresses(count), scalarStorage(count);
  std::vector<void *> packed(count);
  std::vector<std::shared_ptr<Buffer>> buffers;
  for (size_t i = 0; i < count; ++i) {
    if (p->bufferBytes[i]) {
      auto buffer = nb::cast<std::shared_ptr<Buffer>>(arguments[i]);
      if (buffer->device != d || buffer->buffer.size() < p->bufferBytes[i])
        throw std::invalid_argument("buffer device or size does not match compiled signature");
      addresses[i] = buffer->buffer.address();
      packed[i] = &addresses[i];
      buffers.push_back(std::move(buffer));
    } else {
      const auto &type = p->scalarTypes[i];
      auto store = [&](auto value) { std::memcpy(&scalarStorage[i], &value, sizeof(value)); };
      if (type == "i32") store(nb::cast<int32_t>(arguments[i]));
      else if (type == "i64") store(nb::cast<int64_t>(arguments[i]));
      else if (type == "f32") store(nb::cast<float>(arguments[i]));
      else if (type == "f64") store(nb::cast<double>(arguments[i]));
      else throw std::invalid_argument("unsupported scalar binding type");
      packed[i] = &scalarStorage[i];
    }
  }
  nb::gil_scoped_release release;
  // Scalarized intermediates change ordering between threads when external
  // ranges overlap (including shifted views). Keep the original whole-kernel
  // boundaries for those bindings. Compare differences to avoid overflow.
  if (p->aliasFallback) {
    bool overlap = false;
    for (uint64_t i = 0; i < count; ++i)
      for (uint64_t j = i + 1; j < count; ++j) {
        if (!p->bufferBytes[i] || !p->bufferBytes[j])
          continue;
        overlap |= addresses[i] <= addresses[j] ? addresses[j] - addresses[i] < p->bufferBytes[i]
                                                : addresses[i] - addresses[j] < p->bufferBytes[j];
      }
    if (overlap)
      p = p->aliasFallback;
  }
  auto invocation = std::make_shared<Invocation>(p, d);
  invocation->buffers = std::move(buffers);
  p->entry(invocation.get(), packed.data());
  if (!invocation->error.empty())
    throw std::runtime_error(invocation->error);
  return invocation;
}

void runInvocation(Invocation &f, int mode, uint64_t batch) {
  std::lock_guard<std::mutex> lock(f.runMutex);
  validateExecution(f, mode, batch);
  execute(f, mode, batch);
  f.stream.synchronize();
}

std::array<double, 3> benchmarkInvocation(Invocation &f, int mode, uint64_t iterations,
                                       uint64_t batch) {
  std::array<double, 3> result{};
  if (!iterations)
    throw std::invalid_argument("benchmark iterations must be positive");
  std::lock_guard<std::mutex> lock(f.runMutex);
  validateExecution(f, mode, batch);
  using Clock = std::chrono::steady_clock;
  f.stream.synchronize();
  auto setupStart = Clock::now();
  if (mode)
    executionGraph(f, mode, batch);
  result[2] = std::chrono::duration<double, std::milli>(Clock::now() - setupStart).count();
  auto check = [](CUresult error) {
    if (error != CUDA_SUCCESS) {
      const char *message = nullptr;
      cuGetErrorString(error, &message);
      throw std::runtime_error(message ? message : "CUDA event measurement failed");
    }
  };
  struct Event {
    CUevent value = nullptr;
    ~Event() {
      if (value)
        cuEventDestroy(value);
    }
  } start, stop;
  check(cuEventCreate(&start.value, CU_EVENT_DEFAULT));
  check(cuEventCreate(&stop.value, CU_EVENT_DEFAULT));
  auto stream = reinterpret_cast<CUstream>(f.stream.nativeHandle());
  check(cuEventRecord(start.value, stream));
  auto wallStart = Clock::now();
  for (uint64_t i = 0; i < iterations; ++i) {
    execute(f, mode, batch);
  }
  check(cuEventRecord(stop.value, stream));
  check(cuEventSynchronize(stop.value));
  auto wallStop = Clock::now();
  float elapsed;
  check(cuEventElapsedTime(&elapsed, start.value, stop.value));
  result[0] = elapsed * 1000.0 / iterations / batch;
  result[1] = std::chrono::duration<double, std::micro>(wallStop - wallStart).count() /
              iterations / batch;
  return result;
}

// Use Python's contiguous buffer protocol for host transfers, including NumPy and memoryviews.
void copyBuffer(Buffer &buffer, nb::handle object, bool toDevice) {
  Py_buffer view{};
  if (PyObject_GetBuffer(object.ptr(), &view, toDevice ? PyBUF_CONTIG_RO : PyBUF_CONTIG) != 0)
    throw nb::python_error();
  struct Release { Py_buffer *view; ~Release() { PyBuffer_Release(view); } } cleanup{&view};
  if (size_t(view.len) != buffer.buffer.size())
    throw std::invalid_argument("host buffer byte count does not match device storage");
  nb::gil_scoped_release release;
  if (toDevice) buffer.buffer.copyFromHost(view.buf, view.len);
  else buffer.buffer.copyToHost(view.buf, view.len);
}
#endif
} // namespace

NB_MODULE(_ckl_native, m) {
  m.doc() = "CKL in-process compiler and CUDA runtime";
  nb::class_<Program>(m, "Program")
      .def("ir", [](const Program &p, unsigned stage) {
        switch (stage) {
        case 0: return p.preparedIR;
        case 1: return p.executableIR;
        case 2: return p.deviceIR;
        case 3: return p.hostIR;
        default: throw std::invalid_argument("invalid IR stage");
        }
      });
  m.def("compile", [](const std::string &source, const std::string &chip,
                       const std::string &features, const std::string &toolkit,
                       bool fusion, bool reuse) {
    return compile(source.c_str(), chip.c_str(), features.c_str(), toolkit.c_str(), fusion, reuse);
  }, nb::call_guard<nb::gil_scoped_release>());
#ifdef CKL_HAS_CUDA_RUNTIME
  m.attr("has_cuda_runtime") = true;
  nb::class_<Device>(m, "Device")
      .def(nb::init<>())
      .def_prop_ro("name", [](const Device &d) { return d.runtime.deviceName(); })
      .def("empty", [](std::shared_ptr<Device> d, uint64_t bytes) {
        return std::make_shared<Buffer>(Buffer{d, d->runtime.allocate(bytes)});
      })
      .def("import_buffer", [](std::shared_ptr<Device> d, uint64_t address, uint64_t bytes,
                                nb::handle /* owner */) {
        return std::make_shared<Buffer>(Buffer{d, d->runtime.importBuffer(address, bytes)});
      }, nb::keep_alive<0, 4>());
  nb::class_<Buffer>(m, "Buffer")
      .def_prop_ro("address", [](const Buffer &b) { return b.buffer.address(); })
      .def_prop_ro("nbytes", [](const Buffer &b) { return b.buffer.size(); })
      .def("copy_from_host", [](Buffer &b, nb::handle data) { copyBuffer(b, data, true); })
      .def("copy_to_host", [](Buffer &b, nb::handle data) { copyBuffer(b, data, false); });
  nb::class_<Invocation>(m, "Invocation")
      .def("run", &runInvocation, nb::call_guard<nb::gil_scoped_release>())
      .def("benchmark", &benchmarkInvocation, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("node_count", [](const Invocation &f) { return f.plan.nodes().size(); });
  m.def("bind", &bindProgram);
  m.def("cuda_available", [] {
    int count = 0;
    return cuInit(0) == CUDA_SUCCESS && cuDeviceGetCount(&count) == CUDA_SUCCESS && count > 0;
  });
  m.def("profiler_start", [] { if (cuProfilerStart() != CUDA_SUCCESS)
      throw std::runtime_error("CUDA profiler start failed"); });
  m.def("profiler_stop", [] { if (cuProfilerStop() != CUDA_SUCCESS)
      throw std::runtime_error("CUDA profiler stop failed"); });
#else
  m.attr("has_cuda_runtime") = false;
  m.def("cuda_available", [] { return false; });
#endif
}
