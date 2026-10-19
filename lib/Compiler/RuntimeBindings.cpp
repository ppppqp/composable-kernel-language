#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>
#ifdef CKL_HAS_CUDA_RUNTIME
#include "ckl/Runtime/NvidiaProgram.h"
#endif

namespace nb = nanobind;
#ifdef CKL_HAS_CUDA_RUNTIME
using namespace mlir::ckl::runtime;
using Program = NvidiaProgram;
using Device = NvidiaProgramDevice;
using Buffer = NvidiaProgramBuffer;
using Invocation = NvidiaInvocation;
namespace {
std::shared_ptr<NvidiaInvocation> bindProgram(std::shared_ptr<NvidiaProgram> program,
                                             std::shared_ptr<NvidiaProgramDevice> device,
                                             nb::sequence arguments) {
  if (nb::len(arguments) != program->bufferBytes().size())
    throw std::invalid_argument("incorrect program argument count");
  std::vector<NvidiaProgramArgument> values;
  values.reserve(nb::len(arguments));
  for (size_t i = 0; i < nb::len(arguments); ++i) {
    if (program->bufferBytes()[i]) {
      values.emplace_back(nb::cast<std::shared_ptr<NvidiaProgramBuffer>>(arguments[i]));
      continue;
    }
    const auto &type = program->scalarTypes()[i];
    if (type == "i32") values.emplace_back(nb::cast<int32_t>(arguments[i]));
    else if (type == "i64") values.emplace_back(nb::cast<int64_t>(arguments[i]));
    else if (type == "f32") values.emplace_back(nb::cast<float>(arguments[i]));
    else if (type == "f64") values.emplace_back(nb::cast<double>(arguments[i]));
    else throw std::invalid_argument("unsupported scalar binding type");
  }
  nb::gil_scoped_release release;
  return program->bind(std::move(device), values);
}

// Use Python's contiguous buffer protocol for host transfers, including NumPy and memoryviews.
void copyBuffer(NvidiaProgramBuffer &buffer, nb::handle object, bool toDevice) {
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
} // namespace
#endif

void registerRuntimeBindings(nb::module_ &m) {
#ifdef CKL_HAS_CUDA_RUNTIME
  nb::class_<Program>(m, "Program");
  // Keep the Python JIT owner and optional fallback alive with this wrapper.
  // Bound invocation wrappers in turn retain this wrapper through keep_alive.
  m.def("make_program", [](uintptr_t entry, std::vector<uint64_t> bytes,
                            std::vector<std::string> types, nb::handle /* owner */,
                            std::shared_ptr<Program> fallback) {
    return std::make_shared<NvidiaProgram>(entry, std::move(bytes), std::move(types),
                                           std::move(fallback));
  }, nb::arg("entry"), nb::arg("bytes"), nb::arg("types"), nb::arg("owner"),
     nb::arg("fallback").none(), nb::keep_alive<0, 4>(), nb::keep_alive<0, 5>());
  m.def("runtime_symbols", [] {
    nb::dict symbols;
    for (const auto &symbol : nvidiaProgramRuntimeSymbols())
      symbols[symbol.name] = symbol.address;
    return symbols;
  });
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
      .def("slice", [](const Buffer &b, uint64_t offset, uint64_t length) {
        return std::make_shared<Buffer>(Buffer{b.device, b.buffer.slice(offset, length)});
      }, nb::keep_alive<0, 1>())
      .def("copy_from_host", [](Buffer &b, nb::handle data) { copyBuffer(b, data, true); })
      .def("copy_to_host", [](Buffer &b, nb::handle data) { copyBuffer(b, data, false); });
  nb::class_<Invocation>(m, "Invocation")
      .def("run", &NvidiaInvocation::run, nb::call_guard<nb::gil_scoped_release>())
      .def("benchmark", &NvidiaInvocation::benchmark, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("node_count", [](const Invocation &f) { return f.nodeCount(); });
  m.def("bind", &bindProgram, nb::keep_alive<0, 1>(), nb::keep_alive<0, 3>());
  m.def("cuda_available", &nvidiaDeviceAvailable);
  m.def("profiler_start", &nvidiaProfilerStart);
  m.def("profiler_stop", &nvidiaProfilerStop);
#else
  m.attr("has_cuda_runtime") = false;
  m.def("cuda_available", [] { return false; });
#endif
}
