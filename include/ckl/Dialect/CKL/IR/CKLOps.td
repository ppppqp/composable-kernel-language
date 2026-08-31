#ifndef CKL_DIALECT_CKL_IR_CKLOPS_TD
#define CKL_DIALECT_CKL_IR_CKLOPS_TD

include "CKLTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

// Binds the symbolic extents in a tile's logical space to SSA index values.
def CKL_BindSymbolsOp : CKL_Op<"bind_symbols", [Pure]> {
  let summary = "Bind symbolic tile extents to SSA values";
  let arguments = (ins CKL_TileType:$input, Variadic<Index>:$bindings,
                       StrArrayAttr:$symbols);
  let results = (outs CKL_TileType:$result);
  let assemblyFormat = "$input `[` $bindings `]` `symbols` `=` $symbols attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

// A task names a layout-polymorphic callable contract and its implementation
// alternatives. Alternative dictionaries must contain a unique string `name`.
def CKL_TaskOp : CKL_Op<"task", [Symbol]> {
  let summary = "Declare a layout-polymorphic task";
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       ArrayAttr:$alternatives);
  let assemblyFormat = "$sym_name `:` $function_type `alternatives` `=` $alternatives attr-dict";
  let hasVerifier = 1;
}

def CKL_InvokeOp : CKL_Op<"invoke"> {
  let summary = "Invoke a task, optionally selecting an implementation alternative";
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$inputs,
                       OptionalAttr<StrAttr>:$alternative);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$callee `(` $inputs `)` (`alternative` `=` $alternative^)? attr-dict `:` functional-type($inputs, $results)";
  let hasVerifier = 1;
}

def CKL_LoadTileOp : CKL_Op<"load_tile"> {
  let summary = "Load a distributed logical tile from a ranked memref";
  let arguments = (ins AnyMemRef:$source, Variadic<Index>:$offsets,
                       CKL_DistributionAttr:$distribution);
  let results = (outs CKL_TileType:$result);
  let assemblyFormat = "$source `[` $offsets `]` `distribution` `=` qualified($distribution) `attributes` attr-dict `:` type($source) `->` type($result)";
  let hasVerifier = 1;
}

def CKL_StoreTileOp : CKL_Op<"store_tile"> {
  let summary = "Store a distributed logical tile to a ranked memref";
  let arguments = (ins CKL_TileType:$value, AnyMemRef:$target,
                       Variadic<Index>:$offsets,
                       CKL_DistributionAttr:$distribution);
  let assemblyFormat = "$value `,` $target `[` $offsets `]` `distribution` `=` qualified($distribution) `attributes` attr-dict `:` type($value) `,` type($target)";
  let hasVerifier = 1;
}

// An unresolved edge between a producer task and a consumer task. Selection
// replaces this with ckl.compose using the chosen alternatives' port layouts.
def CKL_TaskComposeOp : CKL_Op<"task_compose", [Pure, SameOperandsAndResultType]> {
  let summary = "Jointly select producer and consumer task alternatives";
  let arguments = (ins CKL_TileType:$input,
                       FlatSymbolRefAttr:$producer,
                       StrAttr:$producer_port,
                       FlatSymbolRefAttr:$consumer,
                       StrAttr:$consumer_port,
                       DefaultValuedAttr<I64Attr, "64">:$subgroup_size,
                       DefaultValuedAttr<I64Attr, "256">:$register_limit,
                       DefaultValuedAttr<I64Attr, "65536">:$shared_memory_limit,
                       StrArrayAttr:$capabilities);
  let results = (outs CKL_TileType:$result);
  let assemblyFormat = "$input `producer` `=` $producer `[` $producer_port `]` `consumer` `=` $consumer `[` $consumer_port `]` `capabilities` `=` $capabilities `attributes` attr-dict `:` type($input)";
  let hasVerifier = 1;
}

// Represents an unresolved producer/consumer boundary
def CKL_ComposeOp : CKL_Op<"compose", [Pure, SameOperandsAndResultType]> {
  let summary = "An unresolved producer/consumer composition boundary";
  let arguments = (ins CKL_TileType:$input, CKL_DistributionAttr:$source,
                       CKL_DistributionAttr:$target,
                       DefaultValuedAttr<I64Attr, "64">:$subgroup_size);
  let results = (outs CKL_TileType:$result);
  let assemblyFormat = "$input `source` `=` qualified($source) `target` `=` qualified($target) `attributes` attr-dict `:` type($input)";
  let hasVerifier = 1;
}

// Represents a resolved conversion with a known conversion kind
def CKL_ConvertLayoutOp : CKL_Op<"convert_layout", [Pure, SameOperandsAndResultType]> {
  let summary = "An explicit semantic layout conversion";
  let arguments = (ins CKL_TileType:$input, CKL_DistributionAttr:$source,
                       CKL_DistributionAttr:$target, StrAttr:$kind,
                       DefaultValuedAttr<I64Attr, "64">:$subgroup_size,
                       OptionalAttr<StrAttr>:$reason,
                       OptionalAttr<I64Attr>:$move_count);
  let results = (outs CKL_TileType:$result);
  let assemblyFormat = "$input `source` `=` qualified($source) `target` `=` qualified($target) `kind` `=` $kind `attributes` attr-dict `:` type($input)";
  let hasVerifier = 1;
}

class CKL_ScheduledConversionOp<string mnemonic>
    : CKL_Op<mnemonic, [Pure, SameOperandsAndResultType]> {
  let summary = "A scheduled layout conversion";
  let arguments = (ins CKL_TileType:$input, CKL_DistributionAttr:$source,
                       CKL_DistributionAttr:$target,
                       DefaultValuedAttr<I64Attr, "64">:$subgroup_size,
                       OptionalAttr<StrAttr>:$reason,
                       OptionalAttr<I64Attr>:$move_count);
  let results = (outs CKL_TileType:$result);
  let assemblyFormat = "$input `source` `=` qualified($source) `target` `=` qualified($target) `attributes` attr-dict `:` type($input)";
  let hasVerifier = 1;
}

def CKL_LocalPermuteOp : CKL_ScheduledConversionOp<"local_permute">;
def CKL_SubgroupExchangeOp : CKL_ScheduledConversionOp<"subgroup_exchange">;

class CKL_ExchangePhaseOp<string mnemonic>
    : CKL_Op<mnemonic, [SameOperandsAndResultType]> {
  let summary = "One explicit phase of a scheduled layout exchange";
  let arguments = (ins CKL_TileType:$input, CKL_DistributionAttr:$source,
                       CKL_DistributionAttr:$target,
                       DefaultValuedAttr<I64Attr, "64">:$subgroup_size,
                       OptionalAttr<StrAttr>:$reason,
                       OptionalAttr<I64Attr>:$move_count);
  let results = (outs CKL_TileType:$result);
  let assemblyFormat = "$input `source` `=` qualified($source) `target` `=` qualified($target) `attributes` attr-dict `:` type($input)";
  let hasVerifier = 1;
}

def CKL_SharedStoreOp : CKL_ExchangePhaseOp<"shared_store">;
def CKL_WorkgroupBarrierOp : CKL_ExchangePhaseOp<"workgroup_barrier">;
def CKL_SharedLoadOp : CKL_ExchangePhaseOp<"shared_load">;
def CKL_GlobalStoreOp : CKL_ExchangePhaseOp<"global_store">;
def CKL_KernelBoundaryOp : CKL_ExchangePhaseOp<"kernel_boundary">;
def CKL_GlobalLoadOp : CKL_ExchangePhaseOp<"global_load">;

#endif
