#ifndef CKL_RUNTIME_NVIDIAPROGRAM_H
#define CKL_RUNTIME_NVIDIAPROGRAM_H

#include "ckl/Runtime/NvidiaRuntime.h"

#include <array>
#include <variant>

namespace mlir::ckl::runtime {
namespace detail {
struct NvidiaProgramState;
struct NvidiaInvocationState;
} // namespace detail

/// Shared device identity used to validate buffers and cache loaded modules.
struct NvidiaProgramDevice {
  NvidiaProgramDevice();
  NvidiaRuntime runtime;
  const uint64_t id;
};

struct NvidiaProgramBuffer {
  std::shared_ptr<NvidiaProgramDevice> device;
  NvidiaBuffer buffer;
};

using NvidiaProgramArgument =
    std::variant<std::shared_ptr<NvidiaProgramBuffer>, int32_t, int64_t, float, double>;

class NvidiaInvocation {
public:
  ~NvidiaInvocation();
  /// Modes: 0 ordinary, 1 inferred graph, 2 single-stream capture.
  void run(int mode, uint64_t batch);
  /// Stream us/step, wall us/step, graph setup ms. Synchronizes before returning.
  std::array<double, 3> benchmark(int mode, uint64_t iterations, uint64_t batch);
  std::size_t nodeCount() const;

private:
  friend class NvidiaProgram;
  explicit NvidiaInvocation(std::unique_ptr<detail::NvidiaInvocationState> state);
  std::unique_ptr<detail::NvidiaInvocationState> state_;
};

/// Runtime metadata for a compiled packed host entry. The caller must retain
/// its machine code owner, including fallback code, through every invocation.
/// No MLIR or Python dependency is required by this interface.
class NvidiaProgram {
public:
  NvidiaProgram(uintptr_t entry, std::vector<uint64_t> bufferBytes,
                std::vector<std::string> scalarTypes,
                std::shared_ptr<NvidiaProgram> fallback = {});
  const std::vector<uint64_t> &bufferBytes() const;
  const std::vector<std::string> &scalarTypes() const;
  std::shared_ptr<NvidiaInvocation>
  bind(std::shared_ptr<NvidiaProgramDevice> device,
       const std::vector<NvidiaProgramArgument> &arguments) const;

private:
  std::shared_ptr<detail::NvidiaProgramState> state_;
};

struct NvidiaRuntimeSymbol {
  const char *name;
  uintptr_t address;
};

/// Native callbacks consumed by the ckl_exec-to-LLVM host ABI.
std::array<NvidiaRuntimeSymbol, 8> nvidiaProgramRuntimeSymbols();
bool nvidiaDeviceAvailable();
void nvidiaProfilerStart();
void nvidiaProfilerStop();

} // namespace mlir::ckl::runtime
#endif
