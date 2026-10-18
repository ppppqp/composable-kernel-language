"""Independent NumPy decoder: vectorized linear algebra, no CKL execution."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    batch: int = 1
    hidden: int = 64
    heads: int = 4
    layers: int = 2
    ffn: int = 128
    vocab: int = 256
    context: int = 8
    steps: int = 16

    def __post_init__(self):
        if min(vars(self).values()) <= 0 or self.hidden % self.heads or self.head_dim % 2:
            raise ValueError("positive dimensions and an even head dimension dividing hidden are required")
        if self.layers > 16:
            raise ValueError("this small decoder benchmark supports at most 16 layers")

    @property
    def head_dim(self):
        return self.hidden // self.heads

    @property
    def capacity(self):
        return self.context + self.steps


MUTABLE = ("tokens", "position", "history", "kc", "vc", "logits")
EPSILON = np.float32(1e-5)


def rms(x, gain):
    return x * (1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPSILON)) * gain


def step(arrays, config):
    """Mutate all persistent state for exactly one autoregressive decode step."""
    a, c = arrays, config
    pos = int(a["position"][0])
    if not 0 <= pos < c.capacity:
        raise ValueError("position exceeds KV/history capacity")
    x = a["emb"][a["tokens"]].copy()
    cos, sin = a["sincos"][pos, :, 0], a["sincos"][pos, :, 1]

    def rotate(value):
        pairs = value.reshape(c.batch, c.heads, c.head_dim // 2, 2)
        result = np.empty_like(pairs)
        result[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
        result[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
        return result.reshape(c.batch, c.heads, c.head_dim)

    for layer in range(c.layers):
        n = rms(x, a["norm1"][layer])
        q, k = rotate(n @ a["wq"][layer]), rotate(n @ a["wk"][layer])
        v = (n @ a["wv"][layer]).reshape(c.batch, c.heads, c.head_dim)
        a["kc"][layer, :, :, pos, :] = k
        a["vc"][layer, :, :, pos, :] = v
        keys = a["kc"][layer, :, :, :pos + 1, :]
        scores = np.sum(q[:, :, None, :] * keys, axis=-1) * np.float32(c.head_dim ** -0.5)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        values = a["vc"][layer, :, :, :pos + 1, :]
        context = np.sum(probs[..., None] * values, axis=-2).reshape(c.batch, c.hidden)
        residual = x + context @ a["wo"][layer]
        n = rms(residual, a["norm2"][layer])
        gate, up = n @ a["wg"][layer], n @ a["wu"][layer]
        x = residual + (gate / (1 + np.exp(-gate)) * up) @ a["wd"][layer]
    a["logits"][:] = rms(x, a["final_norm"][0]) @ a["lm"][0]
    a["tokens"][:] = a["logits"].argmax(axis=-1).astype(np.int32)
    a["history"][pos] = a["tokens"]
    a["position"][0] += 1


def make_inputs(c, seed=123, *, prompt_seed=None):
    """Create random model weights and reference-prefill an explicit prompt."""
    rng = np.random.default_rng(seed)
    b, h, l, t, d = c.batch, c.hidden, c.layers, c.capacity, c.head_dim
    shapes = dict(emb=(c.vocab, h), norm1=(l, h), norm2=(l, h), final_norm=(1, h),
                  wq=(l, h, h), wk=(l, h, h), wv=(l, h, h), wo=(l, h, h),
                  wg=(l, h, c.ffn), wu=(l, h, c.ffn), wd=(l, c.ffn, h),
                  lm=(1, h, c.vocab))
    arrays = {}
    for name, shape in shapes.items():
        if "norm" in name:
            arrays[name] = (1 + rng.normal(0, 0.02, shape)).astype(np.float32)
        else:
            scale = 0.5 if name == "emb" else 0.5 / np.sqrt(shape[-2])
            arrays[name] = rng.normal(0, scale, shape).astype(np.float32)
    angles = np.arange(t)[:, None] * (10000. ** (-np.arange(0, d, 2) / d))[None, :]
    arrays["sincos"] = np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype(np.float32)
    prompt_rng = rng if prompt_seed is None else np.random.default_rng(prompt_seed)
    prompts = prompt_rng.integers(0, c.vocab, (c.context, b), dtype=np.int32)
    arrays.update(tokens=prompts[0].copy(), position=np.array([0], dtype=np.int32),
                  history=np.full((t, b), -1, dtype=np.int32),
                  kc=np.full((l, b, c.heads, t, d), -12345, dtype=np.float32),
                  vc=np.full((l, b, c.heads, t, d), -12345, dtype=np.float32),
                  logits=np.full((b, c.vocab), -12345, dtype=np.float32))
    # This preparation is outside decode timing. The last prompt token is the
    # input to the first measured step; generated IDs then feed subsequent steps.
    for i in range(c.context - 1):
        step(arrays, c)
        arrays["tokens"][:] = prompts[i + 1]
    return arrays, prompts


def expected(initial, config, steps):
    result = {name: value.copy() if name in MUTABLE else value for name, value in initial.items()}
    for _ in range(steps):
        step(result, config)
    return result
