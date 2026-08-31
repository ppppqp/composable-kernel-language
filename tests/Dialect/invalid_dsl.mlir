// expected-error @+1 {{has duplicate alternative 'same'}}
ckl.task @bad : (!ckl.tile<f32, #ckl.space<[m = symbol<"M">]>>) -> !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>> alternatives = [{name = "same"}, {name = "same"}]

// -----

#memory_distribution = #ckl.distribution<
  executors = #ckl.space<[lane = 4]>, local = #ckl.space<[value = 2]>,
  tile = #ckl.space<[x = 8]>,
  ownership = #ckl.index_map<domain = #ckl.space<[lane = 4, value = 2]>, codomain = #ckl.space<[x = 8]>, results = [add(mul(dim(0), const(2)), dim(1))], predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[value = 2]>, codomain = #ckl.space<[value = 2]>, results = [dim(0)], predicate = true>,
  scope = subgroup, replicated = false>

module {
  func.func @bad_memory_type(%source: memref<8xf16>, %base: index) {
    // expected-error @+1 {{memref and tile element types must match}}
    %tile = ckl.load_tile %source[%base] distribution = #memory_distribution attributes {} : memref<8xf16> -> !ckl.tile<f32, #ckl.space<[x = 8]>>
    return
  }
}

// -----

func.func @bad_bind(
    %tile: !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>>)
    -> !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>> {
  // expected-error @+1 {{requires exactly one binding for each unique symbolic extent}}
  %bound = ckl.bind_symbols %tile [] symbols = [] : !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>> -> !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>>
  return %bound : !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>>
}

// -----

ckl.task @good : (!ckl.tile<f32, #ckl.space<[m = symbol<"M">]>>) -> !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>> alternatives = [{name = "present"}]

func.func @bad_invoke(
    %tile: !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>>)
    -> !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>> {
  // expected-error @+1 {{selects unknown alternative 'missing'}}
  %result = ckl.invoke @good(%tile) alternative = "missing" : (!ckl.tile<f32, #ckl.space<[m = symbol<"M">]>>) -> !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>>
  return %result : !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>>
}

// -----

// expected-error @+1 {{has invalid resource lifetime for 'scratch'}}
ckl.task @bad_lifetime : () -> () alternatives = [{name = "bad", resources = [{name = "scratch", bytes = 4 : i64, begin = 2 : i64, end = 2 : i64}]}]

// -----

// expected-error @+1 {{has unknown effect kind 'teleport'}}
ckl.task @bad_effect : () -> () alternatives = [{name = "bad", effects = [{kind = "teleport", resource = "tile"}]}]

// -----

// expected-error @+1 {{references unknown implementation @missing}}
ckl.task @missing_body : () -> () alternatives = [{name = "bad", implementation = @missing}]

// -----

func.func private @wrong_body(i32)
// expected-error @+1 {{implementation type '(i32) -> ()' does not match task type '() -> ()'}}
ckl.task @mismatched_body : () -> () alternatives = [{name = "bad", implementation = @wrong_body}]
