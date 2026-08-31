#lhs = #ckl.distribution<
  executors = #ckl.space<[lane = 32]>, local = #ckl.space<[kHalf = 2, rowHalf = 2, pair = 2]>, tile = #ckl.space<[m = 16, k = 16]>,
  ownership = #ckl.index_map<domain = #ckl.space<[lane = 32, kHalf = 2, rowHalf = 2, pair = 2]>, codomain = #ckl.space<[m = 16, k = 16]>,
    results = [add(floordiv(dim(0), 4), mul(dim(2), const(8))), add(add(mul(mod(dim(0), 4), const(2)), dim(3)), mul(dim(1), const(8)))], predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[kHalf = 2, rowHalf = 2, pair = 2]>, codomain = #ckl.space<[address = 8]>,
    results = [add(add(add(const(0), mul(dim(0), const(4))), mul(dim(1), const(2))), mul(dim(2), const(1)))], predicate = true>,
  scope = subgroup, replicated = false>
#rhs = #ckl.distribution<
  executors = #ckl.space<[lane = 32]>, local = #ckl.space<[kHalf = 2, pair = 2]>, tile = #ckl.space<[k = 16, n = 8]>,
  ownership = #ckl.index_map<domain = #ckl.space<[lane = 32, kHalf = 2, pair = 2]>, codomain = #ckl.space<[k = 16, n = 8]>,
    results = [add(add(mul(mod(dim(0), 4), const(2)), dim(2)), mul(dim(1), const(8))), floordiv(dim(0), 4)], predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[kHalf = 2, pair = 2]>, codomain = #ckl.space<[address = 4]>,
    results = [add(add(const(0), mul(dim(0), const(2))), mul(dim(1), const(1)))], predicate = true>,
  scope = subgroup, replicated = false>
#acc = #ckl.distribution<
  executors = #ckl.space<[lane = 32]>, local = #ckl.space<[rowHalf = 2, pair = 2]>, tile = #ckl.space<[m = 16, n = 8]>,
  ownership = #ckl.index_map<domain = #ckl.space<[lane = 32, rowHalf = 2, pair = 2]>, codomain = #ckl.space<[m = 16, n = 8]>,
    results = [add(floordiv(dim(0), 4), mul(dim(1), const(8))), add(mul(mod(dim(0), 4), const(2)), dim(2))], predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[rowHalf = 2, pair = 2]>, codomain = #ckl.space<[address = 4]>,
    results = [add(add(const(0), mul(dim(0), const(2))), mul(dim(1), const(1)))], predicate = true>,
  scope = subgroup, replicated = false>

module {
  gpu.module @kernels {
  ckl.task @mma : (!ckl.tile<f16, #ckl.space<[m = 16, k = 16]>>, !ckl.tile<f16, #ckl.space<[k = 16, n = 8]>>, !ckl.tile<f32, #ckl.space<[m = 16, n = 8]>>) -> !ckl.tile<f32, #ckl.space<[m = 16, n = 8]>> alternatives = [{name = "mma-sync-m16n8k16-f16-f32", implementation_id = "nvidia.mma.sync.m16n8k16.row.col.f32.f16.f16.f32"}]

  gpu.func @kernel(%a: memref<16x16xf16>, %b: memref<16x8xf16>,
                   %c: memref<16x8xf32>, %d: memref<16x8xf32>) kernel {
    %zero = arith.constant 0 : index
    %a_tile = ckl.load_tile %a[%zero, %zero] distribution = #lhs attributes {} : memref<16x16xf16> -> !ckl.tile<f16, #ckl.space<[m = 16, k = 16]>>
    %b_tile = ckl.load_tile %b[%zero, %zero] distribution = #rhs attributes {} : memref<16x8xf16> -> !ckl.tile<f16, #ckl.space<[k = 16, n = 8]>>
    %c_tile = ckl.load_tile %c[%zero, %zero] distribution = #acc attributes {} : memref<16x8xf32> -> !ckl.tile<f32, #ckl.space<[m = 16, n = 8]>>
    %result = ckl.invoke @mma(%a_tile, %b_tile, %c_tile) alternative = "mma-sync-m16n8k16-f16-f32" {ckl.implementation_id = "nvidia.mma.sync.m16n8k16.row.col.f32.f16.f16.f32"} : (!ckl.tile<f16, #ckl.space<[m = 16, k = 16]>>, !ckl.tile<f16, #ckl.space<[k = 16, n = 8]>>, !ckl.tile<f32, #ckl.space<[m = 16, n = 8]>>) -> !ckl.tile<f32, #ckl.space<[m = 16, n = 8]>>
    ckl.store_tile %result, %d[%zero, %zero] distribution = #acc attributes {} : !ckl.tile<f32, #ckl.space<[m = 16, n = 8]>>, memref<16x8xf32>
    gpu.return
  }
  }
}
