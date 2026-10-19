#include "ckl/Runtime/NvidiaProgram.h"

#include <fstream>
#include <iostream>
#include <iterator>

using namespace mlir::ckl::runtime;

namespace {
std::vector<char> cubin;
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
template <typename Function> Function symbol(const char *name) {
  for (const auto &entry : nvidiaProgramRuntimeSymbols())
    if (std::string(entry.name) == name)
      return reinterpret_cast<Function>(entry.address);
  throw std::runtime_error("runtime symbol is missing");
}

// Stand in for packed JIT code without linking Python, nanobind or MLIR.
void entry(void **packed) {
  void *frame = *static_cast<void **>(packed[0]);
  auto args = *static_cast<void ***>(packed[1]);
  require(*static_cast<float *>(args[4]) == 1.25f, "f32 packing changed");
  require(*static_cast<double *>(args[5]) == -2.5, "f64 packing changed");
  require(*static_cast<int64_t *>(args[6]) == -1234567890123LL, "i64 packing changed");
  symbol<void (*)(void *, const void *, uint64_t)>("ckl_begin")(
      frame, cubin.data(), cubin.size());
  symbol<void (*)(void *, uint64_t, uint64_t)>("ckl_heap")(frame, 0, 128);
  auto resource = symbol<void (*)(void *, uint64_t, uint64_t, uint64_t, void *)>("ckl_resource");
  for (uint64_t i = 0; i < 3; ++i)
    resource(frame, i, 0, 0, args[i]);
  resource(frame, 3, 0, 0, nullptr);
  auto kernel = symbol<void (*)(void *, const char *, uint64_t, uint64_t, uint64_t,
                               uint64_t, uint64_t, uint64_t, uint64_t)>("ckl_kernel");
  auto argument = symbol<void (*)(void *, uint64_t)>("ckl_argument_resource");
  for (uint64_t id : {2, 5}) {
    kernel(frame, "join", 1, 1, 1, 32, 1, 1, 0);
    argument(frame, id == 2 ? 0 : 3);
    argument(frame, 1);
    argument(frame, id == 2 ? 3 : 2);
    symbol<void (*)(void *, const void *, uint64_t)>("ckl_argument_value")(
        frame, args[3], sizeof(int32_t));
    if (id == 5)
      symbol<void (*)(void *, uint64_t)>("ckl_dependency")(frame, 2);
    symbol<void (*)(void *, uint64_t)>("ckl_commit")(frame, id);
  }
}
void failingEntry(void **packed) {
  void *frame = *static_cast<void **>(packed[0]);
  auto argument = symbol<void (*)(void *, uint64_t)>("ckl_argument_resource");
  argument(frame, 999); // Records an error rather than unwinding through JIT code.
  argument(frame, 0);   // Subsequent callbacks must remain no-ops.
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (!nvidiaDeviceAvailable())
      return 77;
    require(argc == 2, "expected a validation CUBIN");
    std::ifstream file(argv[1], std::ios::binary);
    cubin.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    require(!cubin.empty(), "could not read validation CUBIN");
    auto device = std::make_shared<NvidiaProgramDevice>();
    auto allocate = [&] {
      return std::make_shared<NvidiaProgramBuffer>(
          NvidiaProgramBuffer{device, device->runtime.allocate(128)});
    };
    auto first = allocate(), second = allocate(), output = allocate();
    std::vector<float> x(32, 1.0f), y(32, 3.0f), result(32);
    first->buffer.copyFromHost(x.data(), 128);
    second->buffer.copyFromHost(y.data(), 128);
    std::vector<uint64_t> sizes{128, 128, 128, 0, 0, 0, 0};
    std::vector<std::string> types{"", "", "", "i32", "f32", "f64", "i64"};
    NvidiaProgram program(reinterpret_cast<uintptr_t>(&entry), sizes, types);
    std::vector<NvidiaProgramArgument> arguments{
        first, second, output, int32_t(32), 1.25f, -2.5, int64_t(-1234567890123LL)};
    auto invocation = program.bind(device, arguments);
    require(invocation->nodeCount() == 2, "dependency plan changed");
    for (int mode : {0, 1, 2}) {
      invocation->run(mode, 2);
      output->buffer.copyToHost(result.data(), 128);
      for (float value : result)
        require(value == 2.21875f, "native program numerical mismatch");
      for (double value : invocation->benchmark(mode, 2, 2))
        require(value >= 0, "invalid native benchmark timing");
    }
    auto wrong = arguments;
    wrong[3] = 32.0f;
    bool rejected = false;
    try { program.bind(device, wrong); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "mismatched scalar type was accepted");
    rejected = false;
    NvidiaProgram failing(reinterpret_cast<uintptr_t>(&failingEntry), sizes, types);
    try { failing.bind(device, arguments); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "callback error was not reported");
    program.bind(device, arguments)->run(1, 1);
    std::cout << "Native program binding, execution and error propagation passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
