"""Small Llama-style decoder. Straightforward f32 kernels, no compute fusion.

The program is one complete autoregressive step. Persistent state (position,
tokens, history, K/V) stays on device across executions. Reductions and dot
products use serial per-thread loops deliberately; this tests orchestration,
not competitive attention/GEMM code generation.
"""
import ckl

B, C, L, A, T, D, F, V, R = [ckl.dim(n) for n in ("B", "C", "L", "A", "T", "D", "F", "V", "R")]
W, I, O = [ckl.dim(n) for n in ("W", "I", "O")]
Hidden = ckl.Buffer[ckl.f32, (B, C)]
Rows = ckl.Buffer[ckl.f32, (B,)]
Tokens = ckl.Buffer[ckl.i32, (B,)]
Position = ckl.Buffer[ckl.i32, (1,)]
History = ckl.Buffer[ckl.i32, (T, B)]
Embedding = ckl.Buffer[ckl.f32, (V, C)]
Gain = ckl.Buffer[ckl.f32, (L, C)]
FinalGain = ckl.Buffer[ckl.f32, (1, C)]
BlockWeight = ckl.Buffer[ckl.f32, (L, C, C)]
WideWeight = ckl.Buffer[ckl.f32, (L, C, F)]
DownWeight = ckl.Buffer[ckl.f32, (L, F, C)]
LMWeight = ckl.Buffer[ckl.f32, (1, C, V)]
Cache = ckl.Buffer[ckl.f32, (L, B, A, T, D)]
SinCos = ckl.Buffer[ckl.f32, (T, R, 2)]
Scores = ckl.Buffer[ckl.f32, (B, A, T)]
Logits = ckl.Buffer[ckl.f32, (B, V)]
Input = ckl.Buffer[ckl.f32, (B, I)]
Output = ckl.Buffer[ckl.f32, (B, O)]
Matrix = ckl.Buffer[ckl.f32, (W, I, O)]


@ckl.kernel
def embedding(tokens: Tokens, table: Embedding, dst: Hidden) -> None:
    c = ckl.global_id(0)
    b = ckl.global_id(1)
    if c < C:
        token = ckl.cast(tokens[b], ckl.index)
        dst[b, c] = table[token, c]


@ckl.kernel
def rms_stats(x: Hidden, inv: Rows, epsilon: ckl.f32) -> None:
    b = ckl.global_id(0)
    if b < B:
        total = 0.0
        for c in range(C):
            total = total + x[b, c] * x[b, c]
        inv[b] = ckl.rsqrt(total / C + epsilon)


@ckl.kernel
def rms_apply(x: Hidden, inv: Rows, gain: Gain, dst: Hidden,
              layer: ckl.constexpr[ckl.index]) -> None:
    c = ckl.global_id(0)
    b = ckl.global_id(1)
    if c < C:
        dst[b, c] = x[b, c] * inv[b] * gain[layer, c]


@ckl.kernel
def linear(x: Input, weight: Matrix, dst: Output, layer: ckl.constexpr[ckl.index]) -> None:
    o = ckl.global_id(0)
    b = ckl.global_id(1)
    if o < O:
        total = 0.0
        for i in range(I):
            total = total + x[b, i] * weight[layer, i, o]
        dst[b, o] = total


@ckl.kernel
def rope(src: Hidden, table: SinCos, position: Position, dst: Hidden) -> None:
    r = ckl.global_id(0)
    h = ckl.global_id(1)
    b = ckl.global_id(2)
    if r < R:
        p = ckl.cast(position[0], ckl.index)
        c = h * (2 * R) + 2 * r
        first = src[b, c]
        second = src[b, c + 1]
        cosine = table[p, r, 0]
        sine = table[p, r, 1]
        dst[b, c] = first * cosine - second * sine
        dst[b, c + 1] = first * sine + second * cosine


@ckl.kernel
def append_kv(k: Hidden, v: Hidden, kc: Cache, vc: Cache, position: Position,
              layer: ckl.constexpr[ckl.index]) -> None:
    d = ckl.global_id(0)
    h = ckl.global_id(1)
    b = ckl.global_id(2)
    if d < D:
        p = ckl.cast(position[0], ckl.index)
        kc[layer, b, h, p, d] = k[b, h * D + d]
        vc[layer, b, h, p, d] = v[b, h * D + d]


@ckl.kernel
def attention_scores(q: Hidden, kc: Cache, position: Position, scores: Scores,
                     scale: ckl.f32, layer: ckl.constexpr[ckl.index]) -> None:
    t = ckl.global_id(0)
    h = ckl.global_id(1)
    b = ckl.global_id(2)
    end = ckl.cast(position[0], ckl.index) + 1
    if t < end:
        total = 0.0
        for d in range(D):
            total = total + q[b, h * D + d] * kc[layer, b, h, t, d]
        scores[b, h, t] = total * scale


@ckl.kernel
def softmax(scores: Scores, position: Position, probs: Scores) -> None:
    b = ckl.global_id(0)
    h = ckl.global_id(1)
    if b < B:
        end = ckl.cast(position[0], ckl.index) + 1
        maximum = scores[b, h, 0]
        for t in range(1, end):
            value = scores[b, h, t]
            maximum = ckl.select(value > maximum, value, maximum)
        total = 0.0
        for t in range(end):
            total = total + ckl.exp(scores[b, h, t] - maximum)
        for t in range(end):
            probs[b, h, t] = ckl.exp(scores[b, h, t] - maximum) / total


@ckl.kernel
def attention_values(probs: Scores, vc: Cache, position: Position, dst: Hidden,
                     layer: ckl.constexpr[ckl.index]) -> None:
    d = ckl.global_id(0)
    h = ckl.global_id(1)
    b = ckl.global_id(2)
    if d < D:
        end = ckl.cast(position[0], ckl.index) + 1
        total = 0.0
        for t in range(end):
            total = total + probs[b, h, t] * vc[layer, b, h, t, d]
        dst[b, h * D + d] = total


@ckl.kernel
def add(x: Hidden, y: Hidden, dst: Hidden) -> None:
    c = ckl.global_id(0)
    b = ckl.global_id(1)
    if c < C:
        dst[b, c] = x[b, c] + y[b, c]


@ckl.kernel
def silu(src: Input, dst: Input) -> None:
    i = ckl.global_id(0)
    b = ckl.global_id(1)
    if i < I:
        value = src[b, i]
        dst[b, i] = value / (1.0 + ckl.exp(-value))


@ckl.kernel
def multiply(x: Input, y: Input, dst: Input) -> None:
    i = ckl.global_id(0)
    b = ckl.global_id(1)
    if i < I:
        dst[b, i] = x[b, i] * y[b, i]


@ckl.kernel
def greedy(logits: Logits, tokens: Tokens, history: History, position: Position) -> None:
    b = ckl.global_id(0)
    if b < B:
        maximum = logits[b, 0]
        winner = ckl.cast(b, ckl.i32) * 0
        for v in range(1, V):
            score = logits[b, v]
            wins = score > maximum
            winner = ckl.select(wins, ckl.cast(v, ckl.i32), winner)
            maximum = ckl.select(wins, score, maximum)
        p = ckl.cast(position[0], ckl.index)
        tokens[b] = winner
        history[p, b] = winner


@ckl.kernel
def advance(position: Position) -> None:
    position[0] = position[0] + 1


@ckl.program
def decode(tokens: Tokens, position: Position, history: History,
           emb: Embedding, norm1: Gain, norm2: Gain, final_norm: FinalGain,
           wq: BlockWeight, wk: BlockWeight, wv: BlockWeight, wo: BlockWeight,
           wg: WideWeight, wu: WideWeight, wd: DownWeight, lm: LMWeight,
           sincos: SinCos, kc: Cache, vc: Cache, logits: Logits,
           epsilon: ckl.f32, attention_scale: ckl.f32) -> None:
    hidden_grid = (ckl.cdiv(C, 64), B)
    row_grid = (ckl.cdiv(B, 32),)
    x = ckl.empty((B, C), dtype=ckl.f32)
    ckl.launch(embedding, tokens, emb, x, grid=hidden_grid, block=(64,))
    for layer in range(L):
        inv = ckl.empty((B,), dtype=ckl.f32)
        normed = ckl.empty_like(x)
        ckl.launch(rms_stats, x, inv, epsilon, grid=row_grid, block=(32,))
        ckl.launch(rms_apply, x, inv, norm1, normed, layer, grid=hidden_grid, block=(64,))
        q = ckl.empty_like(x)
        k = ckl.empty_like(x)
        v = ckl.empty_like(x)
        ckl.launch(linear, normed, wq, q, layer, grid=hidden_grid, block=(64,))
        ckl.launch(linear, normed, wk, k, layer, grid=hidden_grid, block=(64,))
        ckl.launch(linear, normed, wv, v, layer, grid=hidden_grid, block=(64,))
        qr = ckl.empty_like(x)
        kr = ckl.empty_like(x)
        ckl.launch(rope, q, sincos, position, qr, grid=(ckl.cdiv(R, 32), A, B), block=(32,))
        ckl.launch(rope, k, sincos, position, kr, grid=(ckl.cdiv(R, 32), A, B), block=(32,))
        ckl.launch(append_kv, kr, v, kc, vc, position, layer,
                   grid=(ckl.cdiv(D, 32), A, B), block=(32,))
        scores = ckl.empty((B, A, T), dtype=ckl.f32)
        probs = ckl.empty_like(scores)
        ckl.launch(attention_scores, qr, kc, position, scores, attention_scale, layer,
                   grid=(ckl.cdiv(T, 32), A, B), block=(32,))
        ckl.launch(softmax, scores, position, probs, grid=(ckl.cdiv(B, 32), A), block=(32,))
        context = ckl.empty_like(x)
        ckl.launch(attention_values, probs, vc, position, context, layer,
                   grid=(ckl.cdiv(D, 32), A, B), block=(32,))
        projected = ckl.empty_like(x)
        ckl.launch(linear, context, wo, projected, layer, grid=hidden_grid, block=(64,))
        residual = ckl.empty_like(x)
        ckl.launch(add, x, projected, residual, grid=hidden_grid, block=(64,))
        inv2 = ckl.empty((B,), dtype=ckl.f32)
        normed2 = ckl.empty_like(x)
        ckl.launch(rms_stats, residual, inv2, epsilon, grid=row_grid, block=(32,))
        ckl.launch(rms_apply, residual, inv2, norm2, normed2, layer, grid=hidden_grid, block=(64,))
        gate = ckl.empty((B, F), dtype=ckl.f32)
        up = ckl.empty_like(gate)
        ckl.launch(linear, normed2, wg, gate, layer, grid=(ckl.cdiv(F, 64), B), block=(64,))
        ckl.launch(linear, normed2, wu, up, layer, grid=(ckl.cdiv(F, 64), B), block=(64,))
        activated = ckl.empty_like(gate)
        product = ckl.empty_like(gate)
        ckl.launch(silu, gate, activated, grid=(ckl.cdiv(F, 64), B), block=(64,))
        ckl.launch(multiply, activated, up, product, grid=(ckl.cdiv(F, 64), B), block=(64,))
        down = ckl.empty_like(x)
        next_x = ckl.empty_like(x)
        ckl.launch(linear, product, wd, down, layer, grid=hidden_grid, block=(64,))
        ckl.launch(add, residual, down, next_x, grid=hidden_grid, block=(64,))
        x = next_x
    final_inv = ckl.empty((B,), dtype=ckl.f32)
    final_x = ckl.empty_like(x)
    ckl.launch(rms_stats, x, final_inv, epsilon, grid=row_grid, block=(32,))
    ckl.launch(rms_apply, x, final_inv, final_norm, final_x, 0, grid=hidden_grid, block=(64,))
    ckl.launch(linear, final_x, lm, logits, 0, grid=(ckl.cdiv(V, 64), B), block=(64,))
    ckl.launch(greedy, logits, tokens, history, position, grid=row_grid, block=(32,))
    ckl.launch(advance, position, grid=(1,), block=(1,))
