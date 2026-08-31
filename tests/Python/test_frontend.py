import unittest

from ckl import Alternative, IndexType, Module, Space, Symbol, TileType


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


class FrontendTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
