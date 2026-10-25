"""Debug: literal Cython port vs monotonic_core._align_one vs brute force."""
import numpy as np


def literal(value, t_y, t_x, path):
    """Line-by-line transcription of piper's maximum_path_each."""
    max_neg = np.float32(-1e9)
    index = t_x - 1
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg
            else:
                v_cur = value[y - 1, x] if y > 0 else np.float32(0)
            if x == 0:
                if y == 0:
                    v_prev = np.float32(0.0)
                else:
                    v_prev = max_neg
            else:
                v_prev = value[y - 1, x - 1] if y > 0 else np.float32(0)
            value[y, x] = np.float32(value[y, x] + max(v_prev, v_cur))
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or (y > 0 and value[y - 1, index] < value[y - 1, index - 1])):
            index = index - 1
    return value, path


def brute(value, t_y, t_x):
    best = -np.inf
    best_path = None

    def walk(y, x, acc, cols):
        nonlocal best, best_path
        if y == t_y:
            if x == t_x - 1 and acc > best:
                best, best_path = acc, list(cols)
            return
        for nx in (x, x + 1):
            if nx < t_x:
                walk(y + 1, nx, acc + float(value[y, nx]), cols + [nx])
    walk(0, 0, float(value[0, 0]), [0])
    return best, best_path


rng = np.random.default_rng(0)
t_y = int(rng.integers(1, 7))
t_x = int(rng.integers(1, t_y + 1))
orig = rng.normal(size=(t_y, t_x)).astype(np.float32)
print(f"t_y={t_y} t_x={t_x}\norig=\n{orig}")

# literal
v1 = orig.copy()
p1 = np.zeros((t_y, t_x), dtype=np.int32)
literal(v1, t_y, t_x, p1)
cols1 = list(np.argmax(p1, axis=1))
score1 = float(orig[p1.astype(bool)].sum())

# mine
from tts import monotonic_core as mc
v2 = orig.copy()
p2 = np.zeros((t_y, t_x), dtype=np.int32)
mc._align_one(p2, v2, t_y, t_x)
cols2 = list(np.argmax(p2, axis=1))
score2 = float(orig[p2.astype(bool)].sum())

best, best_cols = brute(orig, t_y, t_x)

print("literal  cols:", cols1, "score", score1)
print("mine     cols:", cols2, "score", score2)
print("brute    cols:", best_cols, "score", best)
print("accumulated value literal:\n", v1)
print("accumulated value mine:\n", v2)
print("equal accumulations:", np.allclose(v1, v2, atol=1e-3))
