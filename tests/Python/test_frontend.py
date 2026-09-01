import unittest

from ckl import (
    Alternative,
    Distribution,
    IndexMap,
    IndexType,
    MemRefType,
    Module,
    NVIDIATarget,
    Space,
    Symbol,
    TileType,
    dim,
    func,
    task,
)


def build_copy_module() -> Module:
    module = Module()
    tile = TileType("f32", Space(m=Symbol("M")))
    copy = module.task(
        "copy",
        [tile],
        [tile],
        [Alternative("direct", {"target": "generic"}), Alternative("amd", {"target": "gfx"})],
    )
    function = module.function("bind_and_invoke", [("tile", tile), ("m", IndexType())], [tile])
    bound = function.bind_symbols("bound", function.arg("tile"), {"M": function.arg("m")})
    result = function.invoke("result", copy, [bound], alternative="amd")
    function.return_(result)
    return module


def direct_distribution() -> Distribution:
    executors = Space(lane=4)
    local = Space(value=2)
    tile = Space(x=8)
    return Distribution(
        executors,
        local,
        tile,
        IndexMap(executors.product(local), tile, [dim(0) * 2 + dim(1)]),
        IndexMap(local, Space(address=2), [dim(0)]),
    )


def build_memory_module() -> Module:
    module = Module()
    source_type = MemRefType([8], "f32")
    target_type = MemRefType([None], "f32")
    function = module.function(
        "copy", [("source", source_type), ("target", target_type), ("base", IndexType())]
    )
    tile = function.load_tile(
        "tile", function.arg("source"), [function.arg("base")], direct_distribution()
    )
    function.store_tile(
        tile, function.arg("target"), [function.arg("base")], direct_distribution()
    )
    function.return_()
    return module


class FrontendTest(unittest.TestCase):
    def test_rejects_callable_implementation_with_wrong_signature(self):
        tile = TileType("f32", Space(x=4))

        @func
        def wrong(value: IndexType()) -> tile:
            raise AssertionError("tracing should not reach an invalid implementation")

        with self.assertRaisesRegex(TypeError, "implementation signature does not match"):

            @task(alternatives=[Alternative("wrong", implementation=wrong)])
            def logical(value: tile) -> tile:
                ...

    def test_alternative_serializes_typed_cost_and_capabilities(self):
        alternative = Alternative(
            "candidate",
            {"estimated_execution_cost": 3, "required_capabilities": ["mma"]},
        )
        self.assertIn("estimated_execution_cost = 3 : i64", alternative.mlir())
        self.assertIn('required_capabilities = ["mma"]', alternative.mlir())

    def test_nvidia_target_identity_includes_toolkit(self):
        first = NVIDIATarget("sm_120", "+ptx87", toolkit_root="/cuda/one")
        second = NVIDIATarget("sm_120", "+ptx87", toolkit_root="/cuda/two")
        self.assertNotEqual(first.cache_identity(), second.cache_identity())
        self.assertIn("cubin-chip=sm_120", first.pipeline())

    def test_emits_generic_ckl(self):
        source = build_copy_module().mlir()
        self.assertIn("#ckl.space<[m = symbol<\"M\">]>", source)
        self.assertIn("ckl.task @copy", source)
        self.assertIn('alternative = "amd"', source)
        self.assertNotIn("nvidia", source.lower())

    def test_rejects_wrong_operand_type(self):
        module = Module()
        tile = TileType("f32", Space(m=4))
        task = module.task("copy", [tile], [tile], [Alternative("direct")])
        function = module.function("bad", [("index", IndexType())], [tile])
        with self.assertRaises(TypeError):
            function.invoke("result", task, [function.arg("index")])

    def test_rejects_unknown_pin(self):
        module = Module()
        tile = TileType("f32", Space(m=4))
        task = module.task("copy", [tile], [tile], [Alternative("direct")])
        function = module.function("bad", [("tile", tile)], [tile])
        with self.assertRaises(ValueError):
            function.invoke("result", task, [function.arg("tile")], alternative="missing")

    def test_emits_distribution_and_memory_operations(self):
        source = build_memory_module().mlir()
        self.assertIn("results = [add(mul(dim(0), const(2)), dim(1))]", source)
        self.assertIn("ckl.load_tile", source)
        self.assertIn("ckl.store_tile", source)

    def test_rejects_out_of_range_map_dimension(self):
        with self.assertRaises(ValueError):
            IndexMap(Space(x=4), Space(y=4), [dim(1)])

    def test_affine_divisor_uses_canonical_syntax(self):
        self.assertEqual((dim(0) // 4).mlir(), "floordiv(dim(0), 4)")
        self.assertEqual((dim(0) % 4).mlir(), "mod(dim(0), 4)")
        with self.assertRaises(ValueError):
            dim(0) // dim(1)

    def test_rejects_short_memref(self):
        module = Module()
        function = module.function(
            "short", [("source", MemRefType([4], "f32")), ("base", IndexType())]
        )
        with self.assertRaises(ValueError):
            function.load_tile(
                "tile", function.arg("source"), [function.arg("base")], direct_distribution()
            )


if __name__ == "__main__":
    unittest.main()
