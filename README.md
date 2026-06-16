# DeepBindi CNN models – C inference port

Reference C implementation of all ten CNN architectures defined in
`cnn_models.py`.  The models originate from the following peer-reviewed paper —
please cite it if you use this code or the architectures in your work:

> L. Gutiérrez-Martín, C. López-Ongil, and J. A. Miranda-Calero,
> **"DeepBindi: An End-to-End Fear Detection System Optimized for Extreme-Edge Deployment,"**
> *IEEE Journal of Biomedical and Health Informatics*, vol. 30, no. 1, Jan. 2026.
> DOI: [10.1109/JBHI.2025.3587961](https://doi.org/10.1109/JBHI.2025.3587961)

Three C variants are provided:

| Variant | Directory | Memory strategy | Data type | Use case |
|---------|-----------|----------------|-----------|----------|
| **Dynamic** | `c_port/` | `malloc` / `free` – heap allocated | `float` | Rapid prototyping, host machines |
| **Static** | `static_port/` | Static global pools, no heap | `float` | Embedded targets, no OS |
| **X-HEEP / int32** | `deepbindi_cnn_x_heep/` | Static global pools, no heap | `int32_t` | X-HEEP RISC-V SoC |

The dynamic and static variants are **self-contained** (no dependencies beyond `libc` and `libm`),
produce **identical checksums**, and use the same tensor layout, layer primitives, and compute loops.

The X-HEEP variant implements **three architectures** selectable at compile time
(`-DDEEPBINDI_MODEL=0/1/2`), uses **32-bit integer arithmetic throughout**
(no `float`, no `math.h`), and runs bare-metal on X-HEEP with or without the FPU.

| Model | `-DDEEPBINDI_MODEL` | Weights (ROM) | Act RAM | Description |
|-------|--------------------|-----------|----|-------------|
| CNN_1D_v2 | `0` (default) | 77 KB | 4.8 KB | Baseline: two standard 1D-conv blocks |
| MobileCNN-1D | `1` | 19.4 KB | 6.9 KB | Depthwise-separable blocks (~4× smaller) |
| MobileCNN-SE-1D | `2` | 21.6 KB | 7.2 KB | Depthwise-separable + SE channel attention |

**CNN_1D_v2** is taken directly from the DeepBindi paper above. **MobileCNN-1D** and
**MobileCNN-SE-1D** are new designs for this port, inspired by the MobileNetV1
depthwise-separable factorization (Howard et al., 2017), Squeeze-and-Excitation blocks (Hu et
al., CVPR 2018), and related lightweight CNN literature targeting IoT/embedded devices (e.g.,
Sanjay & Ahmadinia, ICMLA 2019; Nyakuri et al., IEEE Access 2025). The 1D block structure and
int32 quantization scheme are specific to this implementation.

---

## Interactive model tour

A self-contained HTML tutorial walks through every layer of the three X-HEEP models:

```
deepbindi_cnn_x_heep/docs/model_tour.html
```

Open it in any browser (no server required, no internet connection needed). It covers shape
transformations, integer arithmetic formulas, weight counts, MAC counts, and activation-arena
usage at each inference step for CNN_1D_v2, MobileCNN-1D, and MobileCNN-SE-1D.

---

## Quick start

```bash
# Dynamic version (debug build – logging enabled)
cd c_port
make run

# Static version (debug build – logging enabled)
cd static_port
make run

# Production / embedded build (silent, no printf anywhere)
make

# Verify both produce the same numbers
cd static_port
make verify
```

Expected output (same for both):
```
CNN_2D_v1  | shape=(1,1,1,1) | checksum=0.496194 | values=[0.496194]
CNN_2D_v2  | shape=(1,1,1,1) | checksum=0.496471 | values=[0.496471]
CNN_2D_v3  | shape=(1,1,1,1) | checksum=0.498775 | values=[0.498775]
CNN_1D_v1  | shape=(1,1,1,1) | checksum=0.496733 | values=[0.496733]
CNN_1D_v2  | shape=(1,1,1,1) | checksum=0.503341 | values=[0.503341]
CNN_1D_v3  | shape=(1,1,1,1) | checksum=0.502004 | values=[0.502004]
MobileNetV3Custom | shape=(1,1,1,1) | checksum=0.496607 | values=[0.496607]
CNN_1d_tf_sigmoid | shape=(1,1,1,1) | checksum=0.496598 | values=[0.496598]
CNN_1d_tf_softmax | shape=(1,2,1,1) | checksum=1.501474 | values=[0.498526, 0.501474]
CNN_2d_tf_softmax | shape=(1,2,1,1) | checksum=1.511614 | values=[0.488386, 0.511614]
```

Requires: `gcc` (≥ C99) and `libm`. No Python, PyTorch, or TensorFlow needed.

---

## Model catalogue

| ID  | C function                          | Python class                      | Architecture summary                                      |
|-----|-------------------------------------|-----------------------------------|-----------------------------------------------------------|
| PYA | `run_cnn_2d_v1`                     | `CNN_2D_v1`                       | 2D-CNN · 2 conv + 2 FC · binary sigmoid                  |
| PYB | `run_cnn_2d_v2`                     | `CNN_2D_v2`                       | 2D-CNN · 3 parallel branches + 2nd conv + 2 FC           |
| PYC | `run_cnn_2d_v3`                     | `CNN_2D_v3`                       | 2D-CNN · 2 parallel branches + 2nd conv + 2 FC           |
| PYD | `run_cnn_1d_v1`                     | `CNN_1D_v1`                       | 1D-CNN · 1 conv + 1 FC                                   |
| PYE | `run_cnn_1d_v2`                     | `CNN_1D_v2`                       | 1D-CNN · 2 conv + 1 FC                                   |
| PYF | `run_cnn_1d_v3`                     | `CNN_1D_v3`                       | 1D-CNN · 1 conv + 2 FC                                   |
| PYG | `run_mobilenet_v3_custom`           | `MobileNetV3Custom`               | MobileNetV3-Small (width×0.35, 1-ch), 11 MB blocks + SE  |
| TF1 | `run_cnn_1d_tensorflow_sigmoid`     | `CNN_1d_tensorflow_sigmoid`       | Keras 1D-CNN, sigmoid binary output                      |
| TF2 | `run_cnn_1d_tensorflow_softmax`     | `CNN_1d_tensorflow_softmax`       | Keras 1D-CNN, softmax 2-class output                     |
| TF3 | `run_cnn_2d_tensorflow_softmax`     | `CNN_2d_tensorflow_softmax`       | Keras 2D-CNN (1×k kernel), softmax 2-class output        |

### Input conventions

All tensors use **NCHW** layout (batch × channels × height × width).
1-D signals are stored as `(N, C, 1, W)` so the same `conv2d_forward`
primitive handles both 1-D and 2-D cases uniformly.

| Model group         | Input shape (NCHW) | Meaning                             |
|---------------------|--------------------|-------------------------------------|
| 2D-CNN (PYA–PYC)    | (1, 1, 57, 57)     | Single-channel 57×57 feature map    |
| 1D-CNN (PYD–PYF)    | (1, 57, 1, 10)     | 57 features × 10 time steps         |
| MobileNetV3 (PYG)   | (1, 1, 57, 57)     | Single-channel 57×57 feature map    |
| Keras 1D (TF1–TF2)  | (1, 57, 1, 10)     | 57 features × 10 time steps         |
| Keras 2D (TF3)      | (1, 57, 10, 10)    | 57 channels × 10×10 spatial map     |

---

## Directory structure

```
DeepBindi_PortC/
├── cnn_models.py                   Original Python model definitions (PyTorch + Keras)
│
├── c_port/                         ── Dynamic variant (float, all 10 models) ──────────
│   ├── deepbindi_config.h
│   ├── nn_runtime.h/c              Scalar float kernels using malloc/free
│   ├── cnn_models_c.h/c            All 10 model forward passes
│   ├── main.c                      Demo driver: runs all models, prints checksums
│   └── Makefile
│
├── static_port/                    ── Static variant (float, all 10 models) ───────────
│   ├── deepbindi_config.h
│   ├── arena.h/c                   g_weight_pool / g_act_arena bump allocators
│   ├── nn_runtime.h/c              Same float kernels; free/layer_free are no-ops
│   ├── cnn_models_c.h/c            Same models; act_arena_reset() between runs
│   ├── main.c                      Driver; calls arena_stats() after all models
│   └── Makefile
│
└── deepbindi_cnn_x_heep/           ── X-HEEP / int32 variant (3 models) ──────────────
    ├── deepbindi_config.h          TARGET_PC stubs + DEEPBINDI_ENABLE_FPU guard
    ├── arena.h/c                   int32_t static pools (WEIGHT_POOL / ACT_ARENA)
    ├── nn_runtime.h/c              int32_t kernels; no float, no math.h
    │                               Includes batchnorm_rshift_inplace and
    │                               batchnorm_rshift_perchannel for mobile models
    ├── cnn_models_c.h/c            run_cnn_1d_v2()          MODEL=0
    ├── mobile_models_c.h/c         run_mobilecnn_1d()       MODEL=1
    │                               run_mobilecnn_se_1d()    MODEL=2
    ├── main.c                      X-HEEP driver (CSR cycle count, model dispatch)
    ├── test_input.h                Two WEMAC fold-0 samples (NO_FEAR + FEAR)
    ├── weights_cnn_1d_v2.h         CNN_1D_v2 trained weights (19 713 int32, 77 KB)
    ├── weights_mobile1d.h          MobileCNN-1D weights + per-channel shifts (19 KB)
    ├── weights_mobile_se.h         MobileCNN-SE weights + SE shifts (22 KB)
    ├── export_pytorch_weights.py   CNN_1D_v2 .pth -> weights_cnn_1d_v2.h
    ├── export_mobile_weights.py    MobileCNN .pth -> weights_mobile*.h
    │                               (includes data-driven int32 shift calibration)
    ├── extract_test_inputs.py      WEMAC fold-0 samples -> test_input.h
    └── Makefile                    6 targets: 3 models × {dummy, real-weights}
```

---

## Dynamic vs. static – key differences

| Aspect | `c_port/` (dynamic) | `static_port/` (static) |
|--------|---------------------|-------------------------|
| Weight allocation | `malloc` inside `*_layer_create` | `weight_alloc` (bump into `g_weight_pool[]`) |
| Activation allocation | `malloc` inside `tensor_create` | `act_alloc` (bump into `g_act_arena[]`) |
| Freeing | `free` in `tensor_free` / `*_layer_free` | **no-op** – arena is bulk-reset |
| Between-model cleanup | Each tensor freed after use | `act_arena_reset()` at start of each model |
| Static RAM (all 10 models) | OS heap (invisible) | ≈ 10 MB BSS (`g_weight_pool` + `g_act_arena`) |
| OS / libc requirement | `malloc` / `free` | No heap; `stdio` only when logging is on |
| Logging / output | `make run` (debug build) | `make run` (debug build) |
| Embedded / bare-metal ready | No (needs heap) | **Yes** |

The static version prints a memory usage table at the end:

```
── Arena usage ──────────────────────────────────────────
  Weight pool : 1748492 / 2000000 floats  (6830 / 7813 KB)
  Act arena   : 372770  / 500000  floats HWM  (1456 / 1953 KB)
  Tensor pool : 5 / 128 structs
─────────────────────────────────────────────────────────
```

Use this to right-size `WEIGHT_POOL_FLOATS` and `ACT_ARENA_FLOATS` in
`static_port/arena.h` when deploying a single model.

---

## Embedded portability

Both ports are designed to run on bare-metal targets (no OS, no heap).
All platform-specific behaviour is isolated in **`deepbindi_config.h`**,
which is the only file that needs to change per target.

### Logging (silent by default)

By default the build is **fully silent** — no `printf` or `fprintf` anywhere.
Enable output for debugging with:

```bash
make run      # debug build (adds -DDEEPBINDI_ENABLE_LOGGING automatically)
make debug    # same, without running
make          # production / embedded build – no output, no stdio dependency
```

Or pass the flag directly to your cross-compiler:

```makefile
CFLAGS += -DDEEPBINDI_ENABLE_LOGGING   # host / UART debug build
```

### Fatal error handler

Shape mismatches and pool overflows call `DEEPBINDI_FATAL(msg)`.  The default
behaviour depends on the build mode:

| Build | Default behaviour |
|-------|------------------|
| With logging | Print message to stderr + `exit(1)` |
| Without logging | Infinite loop `for(;;){}` → triggers watchdog reset |

Override for your target by defining `DEEPBINDI_FATAL` before the build:

```c
/* ARM Cortex-M: halt at breakpoint */
#define DEEPBINDI_FATAL(msg)  do { __BKPT(0); for(;;){} } while(0)

/* RISC-V: illegal instruction trap */
#define DEEPBINDI_FATAL(msg)  do { __asm__("unimp"); for(;;){} } while(0)

/* Custom UART + watchdog reset */
#define DEEPBINDI_FATAL(msg)  do { uart_puts(msg); system_reset(); } while(0)
```

### Other portability notes

| Issue | `c_port/` / `static_port/` | `deepbindi_cnn_x_heep/` |
|-------|---------------------------|-------------------------|
| `int` width | Shape fields are plain `int`; 16-bit MCUs may overflow on large 2-D tensors. Use a 32-bit toolchain. | Same. |
| Arithmetic type | `float` throughout; no implicit `double` promotion. | `int32_t` throughout; no `float`, no `math.h`. |
| `memset` to zero | Relies on IEEE-754 all-zero = 0.0f. Safe on all common targets. | Uses explicit scalar zero-fill loops; no libc dependency. |
| `%f` format specifier | Not used (newlib-nano limitation). Values printed as scaled integers. | Not used; values are integers, printed with `%d` directly. |
| `%zu` format specifier | Not supported by newlib-nano; `%u` with `(unsigned)` cast used instead. | Same. |
| `stdio.h` / `stdlib.h` | Included only when `DEEPBINDI_ENABLE_LOGGING` is defined. | Always included (logging always on); `stdlib.h` included only with `TARGET_PC`. |
| FPU | Not required; no CSR writes. | Not required for int32 port. Guard with `#ifdef DEEPBINDI_ENABLE_FPU` if adding FP code. |

---

## Layer primitives (`nn_runtime.c` / `static_port/nn_runtime.c`)

| Primitive | Description |
|-----------|-------------|
| `conv2d_forward` | 2-D convolution with padding, stride, groups (incl. depthwise) |
| `batchnorm_forward_inplace` | Inference BN: `γ·(x−μ)/√(σ²+ε) + β` |
| `maxpool2d_forward` | Sliding-window max reduction |
| `adaptive_avg_pool2d_forward` | Global (or partial) average pooling to target H×W |
| `flatten_forward` | Reshape to `(N, C·H·W, 1, 1)` |
| `dense_forward` | Fully-connected: `y = x·Wᵀ + b` |
| `concat_height` | Concatenate two tensors along the H axis |
| `add_forward` | Element-wise addition (residual connections) |
| `channel_scale_forward` | Per-channel scalar multiply (SE attention gate) |
| `relu_inplace` / `sigmoid_inplace` / `softmax_inplace` | Standard activations |
| `hardsigmoid_inplace` / `hardswish_inplace` | MobileNetV3 approximated activations |

### Fused building blocks (`cnn_models_c.c`)

| Helper | Fuses |
|--------|-------|
| `apply_conv_bn_act` | Conv2D → BatchNorm → Activation |
| `apply_dense_bn_act` | Dense → (optional BN) → Activation |
| `apply_se_block` | GlobalAvgPool → FC → ReLU → FC → HardSigmoid → Scale |
| `apply_mobilenet_block` | Expand conv → Depthwise conv → SE → Project conv → (Residual add) |

---

## Design decisions

**Dropout** is a training-only operation. It is a pure identity at inference
time and is omitted entirely.

**BatchNorm** is applied in inference mode (frozen running mean/var).  The
closed-form formula `γ·(x−μ)/√(σ²+ε) + β` is computed directly.  Dummy
parameters are deterministic and reproducible – replace with real trained values
for production.

**1-D convolutions as 2-D**: the Keras models (TF1–TF3) already represent 1-D
convolutions as 2-D kernels of shape `(1×k)`.  The C port adopts the same
representation uniformly, so all convolutions go through `conv2d_forward`.

**MobileNetV3 channel scaling**: `width_mult = 0.35` is applied to every
channel count with a `make_divisible(..., 8)` rounding step (matching
torchvision) to keep memory accesses aligned.

---

## X-HEEP deployment variant (`deepbindi_cnn_x_heep/`)

This directory contains a dedicated port for the
[X-HEEP](https://github.com/esl-epfl/x-heep) RISC-V SoC supporting three
architectures (CNN_1D_v2, MobileCNN-1D, MobileCNN-SE-1D), targeting
bare-metal inference with or without a hardware FPU.

### Why a separate variant?

| Constraint | Source | Implication for C code |
|------------|--------|------------------------|
| 32-bit word length | HW accelerator requirement | `int32_t` throughout; no `int8_t` TFLite quantization |
| No FPU guarantee | X-HEEP bare-metal startup | No `float`; no `math.h` (`expf`, `sqrtf`, `fabsf` banned) |
| No heap allocator | Bare-metal, no OS | Static global pools; `malloc`/`free` banned |
| No `memset` / `memcpy` | Avoid libc symbol dependencies | Explicit scalar loops everywhere |
| No `%f` in `printf` | newlib-nano limitation | Integer printing only |
| CSR cycle counter | X-HEEP hardware performance measurement | `CSR_WRITE/READ(CSR_REG_MCYCLE, ...)` |

### Key design changes vs `static_port/`

**All arithmetic is `int32_t`** — no `float` anywhere in the data path:

```c
typedef struct {
    int      n, c, h, w;
    int32_t *data;          /* was float * */
} Tensor;
```

**BatchNorm is pre-folded to Q7 scale + offset** — eliminates `sqrtf` from the
forward pass entirely:

```c
typedef struct {
    int      num_features;
    int32_t *scale;   /* Q7: scale[c] = round(gamma[c]/sqrt(var[c]+eps) * 128) */
    int32_t *offset;  /* offset[c] = round(beta[c] - gamma[c]*mean[c]/sqrt(var[c]+eps)) */
} BatchNormLayer;

/* Forward: */
y = (int32_t)(((int64_t)x * scale[c]) >> 7) + offset[c];
```

**Sigmoid replaced by a sign threshold** — the output layer is binary (fear / no fear),
so `sigmoid(x) > 0.5` is equivalent to `x > 0`, which requires no `expf`:

```c
void sigmoid_inplace(Tensor *input) {
    for (i = 0; i < total; ++i)
        input->data[i] = (input->data[i] > 0) ? 1 : 0;
}
```

**FPU enable is optional** — the int32 port does not trigger any FP instruction,
so the `mstatus.FS` write is guarded:

```c
#ifdef DEEPBINDI_ENABLE_FPU
    CSR_SET_BITS(CSR_REG_MSTATUS, (FS_INITIAL << 13));
#endif
```

**PC testing** — build with a standard host `gcc` using `-DTARGET_PC` (which
the `Makefile` sets automatically). This stubs all CSR macros and redirects
`DEEPBINDI_FATAL` to `exit(1)`:

```bash
cd deepbindi_cnn_x_heep
make run
```

Expected output (dummy weights, test sample 0):
```
DeepBindi CNN_1D_v2 on X-HEEP
int32 inference, test sample 0 (label=0)
Output : 1 (FEAR)
Cycles : 0
-- Arena usage --
  Weight pool : 19713 / 24000 words  (77 / 93 KB)
  Act arena   : 1211 / 2048 words  (4 / 8 KB)
  Tensor pool : 7 / 16 structs
-----------------
```
Note: with dummy (seeded pseudo-random) weights the output is meaningless — `FEAR`
here does not indicate a real prediction. The arena numbers are the meaningful
check: weight pool usage (19 713/24 000) and act arena (1 211/2 048) must match
these values exactly for any correct build.

### Static SRAM footprint

All three models share the same 2 048-word (8 KB) activation arena.
In the **real-weights build** weights live in `.rodata` (flash); the pool is 1 word.

```
                  ROM (flash)          SRAM at inference
  CNN_1D_v2       77 KB  weights       4.8 KB act arena
  MobileCNN-1D    19 KB  weights       6.9 KB act arena
  MobileCNN-SE    22 KB  weights       7.2 KB act arena
  ─────────────────────────────────────────────────────
  Shared pool                          8.0 KB (ACT_ARENA ceiling)
  Tensor pool                          0.4 KB (16 × sizeof(Tensor))
```

| Buffer | Real-weights build | Dummy-weights build |
|--------|-------------------|---------------------|
| `g_weight_pool[]` | 1 word (4 B) — no-op | 24 000 words for CNN_1D_v2; 6 000 for mobile |
| `g_act_arena[]` | 2 048 words (8 KB) — shared by all models | same |
| Weights | `.rodata` in flash | BSS in `g_weight_pool[]` |

Dummy builds are for PC arena profiling only and should not be flashed to X-HEEP.

### Overflow analysis

| Layer | Max accumulator value | Headroom |
|-------|----------------------|----------|
| Conv1: 57×5 MACs, inputs ≤ 127, weights ≤ 8 | 285 × 127 × 8 ≈ 290 K | INT32_MAX = 2.1 G ✓ |
| Conv2: 32×5 MACs, inputs ≤ 290 K, weights ≤ 8 | 160 × 290 K × 8 ≈ 371 M | INT32_MAX = 2.1 G ✓ |
| BN multiply (int64 intermediate): 371 M × 128 ≈ 47 G | handled by `int64_t` cast | ✓ |
| Dense: 64 MACs, inputs ≤ 371 M, weights ≤ 8 | 64 × 371 M × 8 ≈ 190 G | handled at INT32 post-BN clamp |

In practice, pseudo-random dummy weights cancel out; worst-case values are achieved
only when all weights and inputs have the same sign.

### Test data

`test_input.h` contains sample 0 from
`CH07_TFLite/saved_model/micro/test_data.txt` (label = 0, NO_FEAR):

```c
static const int32_t test_input_0[570] = { 7, 8, 13, 10, ... };
```

Layout: `data[ch * 10 + t]` for channel `ch` ∈ [0, 56], time step `t` ∈ [0, 9].
Values are original int8-range integers widened to `int32_t`.

### Important: TFLite model vs CNN_1D_v2 architecture mismatch

The `.tflite` files in `CH07_TFLite/saved_model/tflite/` are trained weights for
`CNN_2d_tensorflow_softmax` (TF3 / `model_quant_1FC.tflite`), **not** for the
PyTorch CNN_1D_v2 (PYE) that this C port implements:

| | CNN_1D_v2 (this C port) | TFLite micro (model_quant_1FC) |
|---|---|---|
| Input layout | NCHW `(1, 57, 1, 10)` | NHWC `(1, 57, 10, 1)` |
| Conv blocks | 2 — channels 57→32→64 | 1 — filters 1→64 |
| Kernel | `(1×5)` × 2 | `(1×5)` × 1 |
| Flatten features | 64 | 57 × 3 × 64 = 10 944 |
| Output | 1 × threshold(0) | 2-class softmax + argmax |

Consequently the `.tflite` weights **cannot be loaded directly into the C port**.
Use `extract_weights.py` to inspect the TFLite model structure and quantization
parameters. To load real weights into the C port, export the PyTorch CNN_1D_v2
checkpoint instead (see *Replacing dummy weights* below).

### Building for X-HEEP (CMake)

Add the application to the X-HEEP build system and build as usual:

```bash
cmake -DAPP=deepbindi_cnn_x_heep [other X-HEEP flags] ..
make
```

The application directory (`deepbindi_cnn_x_heep/`) is self-contained and
follows the same conventions as other X-HEEP example applications
(`example_matadd`, `example_matfloat`, etc.).

---

## CGRA acceleration guide

A **Coarse-Grained Reconfigurable Array (CGRA)** accelerates computation by
mapping loop nests onto a 2-D array of functional units connected by a
configurable interconnect.  CGRAs excel at **data-parallel, regular loop
structures with predictable memory access patterns** – exactly what neural
network inference provides.

### Primary kernel: `conv2d_forward` ★★★ (highest priority)

Located in `nn_runtime.c`. The 7-level loop nest is:

```c
for n                        // batch – independent per sample
  for oc                     // output channel
    for oh, ow               // spatial output  ← tile across CGRA rows/cols
      sum = bias[oc]
      for icg, kh, kw        // filter window   ← MAC chain on FUs
        sum += input[...] * weight[...]   /* one MAC per iteration */
      output[n][oc][oh][ow] = sum
```

Key observations:

- The innermost `(kh, kw)` loops are **one MAC with no loop-carried
  dependence** across different output positions – the textbook CGRA MAC-chain.
- The `(oh, ow)` loops produce independent output pixels; distribute them
  across CGRA rows/columns as a **spatial tile**.
- **Depthwise convolutions** (MobileNetV3, `groups == in_channels`) collapse the
  `icg` loop to 1, making scheduling simpler with the same MAC structure.
- **BN + activation fusion**: BN is a per-channel scale+shift; ReLU is a
  compare-with-zero. Both can be merged into the CGRA output stage immediately
  after the final accumulate, eliminating two memory round-trips per element.

### Secondary kernel: `dense_forward` ★★

```c
for n               // batch
  for out           // output neuron   ← map across CGRA rows
    sum = bias[out]
    for in          // inner product   ← MAC pipeline per row
      sum += x[in] * W[out][in]
    output[out] = sum
```

FC layers in these models are small (32–192 neurons) – a compact CGRA covers
them without tiling.

### Memory access patterns

| Access | Pattern | CGRA hint |
|--------|---------|-----------|
| Conv weights | Sequential; reused over all `(oh,ow)` | Broadcast / double-buffer |
| Input activations (conv) | Sliding window stencil | Line buffer / shift register |
| Output activations | One write per `(n,oc,oh,ow)` | Direct DMA out |
| Dense weight matrix | Sequential row reads | Sequential DMA |
| BN parameters | One scalar per channel, broadcast over H×W | Constant broadcast |
| SE squeeze vector | One scalar per channel after global avg-pool | Small local buffer |

### Operation count by model

| Model | Dominant ops | CGRA notes |
|-------|-------------|------------|
| PYA (`CNN_2D_v1`) | 2×Conv2D(5×5) + 2×Dense | Simplest 2-D model; good first 2-D test |
| PYB (`CNN_2D_v2`) | 3 parallel Conv2D + Conv2D + 2×Dense | Branches are fully independent (parallelisable) |
| PYC (`CNN_2D_v3`) | 2 parallel Conv2D + Conv2D + 2×Dense | Two-branch variant of PYB |
| PYD (`CNN_1D_v1`) | 1×Conv(1×5) + 1×Dense | **Recommended first CGRA target** |
| PYE (`CNN_1D_v2`) | 2×Conv(1×5) + 1×Dense | Two sequential conv stages |
| PYF (`CNN_1D_v3`) | 1×Conv(1×5) + 2×Dense | Two FC stages |
| PYG (`MobileNetV3`) | 13×pointwise + 11×depthwise + 11×SE + 2×Dense | Most complex; SE adds GlobalAvgPool + 2 small FC per block |
| TF1–TF2 | Conv(1×5) + 2×Dense | Keras equivalents of PYF |
| TF3 | Conv(1×5) + 1×Dense | 2-D kernel emulating 1-D |

### Suggested progression

1. **Start with `run_cnn_1d_v1` (PYD)** – one `conv2d_forward` with a `(1×5)`
   kernel (57 input channels → 64 output, 1-D FIR pattern) plus one
   `dense_forward` (64→1).  Total MACs ≈ 21 900.  Easy to verify.

2. **Scale to `run_cnn_2d_v1` (PYA)** – 2-D spatial tiling over 57×57 feature
   maps.

3. **Tackle `run_mobilenet_v3_custom` (PYG)** – full depthwise + SE pipeline
   with 11 inverted-residual blocks.

### Replacing a primitive with a CGRA implementation

Each primitive has a well-defined C function signature.  To swap in a CGRA
version without touching model code:

1. Implement the same signature in `nn_runtime_cgra.c`.
2. In the Makefile, replace `nn_runtime.c`:

   ```makefile
   SOURCES := main.c nn_runtime_cgra.c cnn_models_c.c      # dynamic
   SOURCES := main.c arena.c nn_runtime_cgra.c cnn_models_c.c  # static
   ```

3. `cnn_models_c.c` and `main.c` are unchanged – they call through the same
   header (`nn_runtime.h`).

For partial acceleration (e.g. only `conv2d_forward`), keep `nn_runtime.c` and
guard with a compile-time flag:

```c
/* nn_runtime_cgra.c */
#include "nn_runtime.h"
Tensor *conv2d_forward(...) { /* CGRA path */ }
/* all other primitives: link against nn_runtime.c for the scalar fallback */
```

---

## Validating CGRA results

Both `c_port/main.c` and `static_port/main.c` print a **checksum** (sum of
absolute output values) for each model.  Use these as reference values:

```bash
# software reference
cd c_port && make run > ref.txt

# after CGRA substitution
make run > cgra.txt

diff ref.txt cgra.txt    # should be identical (or within ~1e-4 tolerance)
```

The `tensor_checksum` helper is defined in `nn_runtime.c`.  For stricter
validation, compare element-wise with a tolerance of `1e-4`.

---

## Replacing dummy weights with real trained weights

### `c_port/` and `static_port/` (float)

1. Export PyTorch weights to a flat binary (`torch.save` + a custom extraction
   script, or ONNX export + `onnx` Python package).
2. Replace the `*_layer_create()` calls in `cnn_models_c.c` with a loader that
   fills pre-allocated `float` arrays from the binary file.
3. For the **static** variant, pre-populate `g_weight_pool[]` at link time using
   a generated C header (`weights_cnn_1d_v2.h`) with trained values as a
   `static const float` array.
4. Verify correctness by comparing `tensor_checksum` against a Python reference
   forward pass on the same input values.

### `deepbindi_cnn_x_heep/` (int32)

The quantized `.tflite` models in `CH07_TFLite/saved_model/tflite/` are for a
**different architecture** (see architecture mismatch note above). To load real
weights into the CNN_1D_v2 C port:

1. **Option A — PyTorch export (recommended):** Load the trained PyTorch
   `CNN_1D_v2` checkpoint, iterate over `model.state_dict()`, quantize each
   weight tensor to int8 range (multiply by a per-layer scale, round, clamp to
   ±127), and write a `weights_cnn_1d_v2.h` header with `const int32_t` arrays.

2. **Option B — TFLite inspection only:** Run `extract_weights.py` to inspect the
   TFLite model and understand its quantization parameters. These weights are not
   directly usable in the C port but are helpful for comparison and cross-validation.

   ```bash
   python deepbindi_cnn_x_heep/extract_weights.py --inspect
   python deepbindi_cnn_x_heep/extract_weights.py  # writes weights_tflite_1FC.h
   ```

3. Once `weights_cnn_1d_v2.h` exists, in `nn_runtime.c` replace the
   `seeded_value_int32()` loops with reads from the const arrays:
   ```c
   /* in conv2d_layer_create: */
   for (i = 0; i < weight_count; ++i)
       layer.weights[i] = cnn1d_conv1_weights[i];
   ```
   The linker places `const` arrays in `.rodata` (flash), reducing SRAM from
   ~94 KB to ~8 KB (activations only).

4. Pre-fold trained BN parameters into Q7 scale+offset:
   ```python
   scale_int  = np.round(gamma / np.sqrt(var + eps) * 128).astype(np.int32)
   offset_int = np.round(beta - gamma * mean / np.sqrt(var + eps)).astype(np.int32)
   ```

---

## Citation

If you use these models or this C port in your work, please cite the original paper:

```bibtex
@article{gutierrez2026deepbindi,
  author    = {Gutiérrez-Martín, Laura and López-Ongil, Celia and Miranda-Calero, Jose A.},
  title     = {{DeepBindi}: An End-to-End Fear Detection System Optimized for Extreme-Edge Deployment},
  journal   = {IEEE Journal of Biomedical and Health Informatics},
  volume    = {30},
  number    = {1},
  year      = {2026},
  doi       = {10.1109/JBHI.2025.3587961},
  note      = {Date of publication: 10 July 2025; date of current version: 8 January 2026}
}
```

**Plain-text reference:**

L. Gutiérrez-Martín, C. López-Ongil, and J. A. Miranda-Calero, "DeepBindi: An End-to-End Fear Detection System Optimized for Extreme-Edge Deployment," *IEEE Journal of Biomedical and Health Informatics*, vol. 30, no. 1, Jan. 2026, doi: 10.1109/JBHI.2025.3587961.

**Context:** The paper presents a fear-recognition system based on physiological signals (BVP, SKT, GSR) from the WEMAC dataset, achieving 80% F1-score and 74% accuracy. The system was validated on an ultra-low-power ARM Cortex-M4 (16 mW @ 5 V, 496 ms per inference). The C port in this repository implements the same model architectures to support deployment on similar extreme-edge targets.
