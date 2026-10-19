"""Exercise the host lowering independently of CUDA compilation and hardware."""

import subprocess
import sys
import unittest

SOURCE = r'''
module attributes {ckl_exec.binary = "ELF\00\FF"} {
  func.func private @step(memref<4xf32>, f32)
  "ckl_exec.plan"() <{source = @step, name = "step.exec", backend = "cuda"}> ({
    "ckl_exec.heap"() <{id = 0 : i64, bytes = 16 : i64, alignment = 16 : i64,
      device = 0 : i64, address_space = "global"}> : () -> ()
    "ckl_exec.resource"() <{name = "arg0", kind = "external", device = "cuda:0",
      address_space = "global", alignment = 16 : i64, bytes = 16 : i64}> : () -> ()
    "ckl_exec.resource"() <{name = "scratch", kind = "temporary", device = "cuda:0",
      address_space = "global", alignment = 16 : i64, bytes = 16 : i64,
      heap = 0 : i64, offset = 0 : i64}> : () -> ()
    "ckl_exec.kernel"() <{id = 2 : i64, kernel = @kernels::@producer,
      implementation = "producer.v1", artifact = "module.v1", abi = "cuda.direct",
      device = "cuda:0", grid = array<i64: 1, 1, 1>, block = array<i64: 4, 1, 1>,
      shared_memory = 0 : i64, dependencies = array<i64>, arguments = [
        {slot = 1 : i64, logical_index = 1 : i64, kind = "scalar", name = "arg1", type = f32},
        {slot = 0 : i64, logical_index = 0 : i64, kind = "resource", resource = "arg0",
          packing = "direct_pointer", type = memref<4xf32>},
        {slot = 2 : i64, logical_index = 2 : i64, kind = "constant", value = -7 : i64, type = i64},
        {slot = 3 : i64, logical_index = 3 : i64, kind = "constant", value = -0.0 : f64, type = f64},
        {slot = 4 : i64, logical_index = 4 : i64, kind = "bytes",
          value = array<i8: 0, -1, 1>, type = vector<3xi8>}
      ]}> : () -> ()
    "ckl_exec.kernel"() <{id = 5 : i64, kernel = @kernels::@consumer,
      implementation = "consumer.v1", artifact = "module.v1", abi = "cuda.direct",
      device = "cuda:0", grid = array<i64: 1, 1, 1>, block = array<i64: 4, 1, 1>,
      shared_memory = 0 : i64, dependencies = array<i64: 2>, arguments = [
        {slot = 0 : i64, logical_index = 0 : i64, kind = "resource", resource = "scratch",
          packing = "direct_pointer", type = memref<4xf32>}
      ]}> : () -> ()
  }) : () -> ()
}
'''

EXECUTABLE = sys.argv.pop(1)


def lower(source):
    return subprocess.run([EXECUTABLE, "-", "--ckl-lower-exec-to-llvm"],
                          input=source, text=True, capture_output=True)


class ExecToLLVMTest(unittest.TestCase):
    def test_runtime_abi_and_exact_bytes(self):
        result = lower(SOURCE)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = result.stdout
        self.assertNotIn("ckl_exec.", text)
        self.assertNotIn("func.func", text)
        self.assertIn("llvm.func @ckl_entry", text)
        for data in (r'ELF\00\FF\00', r'\F9\FF\FF\FF\FF\FF\FF\FF\00',
                     r'\00\00\00\00\00\00\00\80\00', r'\00\FF\01\00'):
            self.assertIn(data, text)
        calls = [line.strip() for line in text.splitlines() if "llvm.call" in line]
        names = [line.split("@")[1].split("(")[0] for line in calls]
        self.assertEqual(names, ["ckl_begin", "ckl_heap", "ckl_resource", "ckl_resource",
                                "ckl_kernel", "ckl_argument_resource",
                                *(["ckl_argument_value"] * 4), "ckl_commit",
                                "ckl_kernel", "ckl_argument_resource", "ckl_dependency", "ckl_commit"])
        self.assertEqual(text.count("llvm.func @ckl_argument_resource"), 1)
        # Lowered modules are valid standalone MLIR and require no CKL dialect ops.
        parsed = subprocess.run([EXECUTABLE, "-"], input=text, text=True, capture_output=True)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)

    def test_rejects_unsupported_or_inconsistent_handoffs(self):
        cases = [
            (SOURCE.replace('backend = "cuda"', 'backend = "hip"'), "requires the CUDA backend"),
            (SOURCE.replace('ckl_exec.binary = "ELF\\00\\FF"', 'ckl_exec.binary = ""'),
             "non-empty ckl_exec.binary"),
            (SOURCE.replace('name = "arg1"', 'name = "arg9"'), "original argN bindings"),
            (SOURCE.replace('name = "arg1"', 'name = "arg0"'), "binding type does not match"),
            (SOURCE.replace('abi = "cuda.direct"', 'abi = "unknown"'), "requires cuda.direct"),
            (SOURCE.replace('device = "cuda:0"', 'device = "cuda:1"'), "requires cuda:0"),
            (SOURCE.replace('func.func private @step(memref<4xf32>, f32)',
                            'func.func private @step(memref<4xf32>, f32) -> i32'),
             "source function declaration with no results"),
            (SOURCE.replace('func.func private @step', 'func.func private @unrelated()\n  func.func private @step'),
             "unexpected operation"),
        ]
        for source, message in cases:
            with self.subTest(message=message):
                result = lower(source)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
