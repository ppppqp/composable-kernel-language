#ifndef CKL_DIALECT_EXEC_IR_CKLEXECOPS_TD
#define CKL_DIALECT_EXEC_IR_CKLEXECOPS_TD

include "ckl/Dialect/Exec/IR/CKLExecDialect.td"
include "mlir/IR/SymbolInterfaces.td"

def CKLExec_PlanOp : CKLExec_Op<"plan", [SingleBlock, NoTerminator,
                                           IsolatedFromAbove]> {
  let summary = "owns a verified backend-ready execution plan";
  let arguments = (ins SymbolRefAttr:$source,
                       StrAttr:$name,
                       StrAttr:$backend);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def CKLExec_HeapOp : CKLExec_Op<"heap", [HasParent<"PlanOp">]> {
  let summary = "declares one device heap allocation";
  let arguments = (ins I64Attr:$id,
                       I64Attr:$bytes,
                       I64Attr:$alignment,
                       I64Attr:$device,
                       StrAttr:$address_space);
  let hasVerifier = 1;
}

def CKLExec_ResourceOp : CKLExec_Op<"resource", [HasParent<"PlanOp">]> {
  let summary = "binds an external resource or a planned heap slice";
  let arguments = (ins StrAttr:$name,
                       StrAttr:$kind,
                       StrAttr:$device,
                       StrAttr:$address_space,
                       I64Attr:$alignment,
                       OptionalAttr<I64Attr>:$bytes,
                       OptionalAttr<I64Attr>:$heap,
                       OptionalAttr<I64Attr>:$offset);
  let hasVerifier = 1;
}

def CKLExec_KernelOp : CKLExec_Op<"kernel", [HasParent<"PlanOp">]> {
  let summary = "launches a kernel with explicit ABI and graph dependencies";
  let arguments = (ins I64Attr:$id,
                       SymbolRefAttr:$kernel,
                       StrAttr:$implementation,
                       StrAttr:$abi,
                       StrAttr:$device,
                       DenseI64ArrayAttr:$grid,
                       DenseI64ArrayAttr:$block,
                       I64Attr:$shared_memory,
                       ArrayAttr:$arguments,
                       DenseI64ArrayAttr:$dependencies);
  let hasVerifier = 1;
  /*
    Example:
    "ckl_exec.kernel"() <{
      id = 4,
      kernel = @kernels::@write,
      implementation = "write.<hash>",
      abi = "cuda.direct",
      device = "cuda:0",
      grid = array<i64: 2, 1, 1>,
      block = array<i64: 256, 1, 1>,
      shared_memory = 0,
      arguments = [
        {
          slot = 0,
          logical_index = 0,
          kind = "resource",
          resource = "temporary0",
          type = memref<257xf32>
        }
      ],
      dependencies = array<i64: 2, 3>
    }>
  */
}

#endif
