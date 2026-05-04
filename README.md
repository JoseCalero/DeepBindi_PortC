# DeepBindi CNN models – C inference port

Reference C implementation of all ten CNN architectures defined in
`cnn_models.py`.  The models originate from the following peer-reviewed paper —
please cite it if you use this code or the architectures in your work:

> L. Gutiérrez-Martín, C. López-Ongil, and J. A. Miranda-Calero,
> **"DeepBindi: An End-to-End Fear Detection System Optimized for Extreme-Edge Deployment,"**
> *IEEE Journal of Biomedical and Health Informatics*, vol. 30, no. 1, Jan. 2026.
> DOI: [10.1109/JBHI.2025.3587961](https://doi.org/10.1109/JBHI.2025.3587961)

Two C variants are provided:

| Variant | Directory | Memory strategy | Use case |
|---------|-----------|----------------|----------|
| **Dynamic** | `c_port/` | `malloc` / `free` – heap allocated | Rapid prototyping, host machines |
| **Static** | `static_port/` | Static global pools, no heap | CGRA / embedded targets, no OS |

Both variants are **self-contained** (no dependencies beyond `libc` and `libm`),
produce **identical checksums**, and use the same tensor layout, layer
primitives, and CGRA-annotated compute loops.

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
model/
├── cnn_models.py           Original Python model definitions (PyTorch + Keras)
│
├── c_port/                 ── Dynamic variant ──────────────────────────────────
│   ├── deepbindi_config.h  Logging + fatal-error macros (embedded portability)
│   ├── nn_runtime.h        Tensor types, layer structs, all prototypes
│   ├── nn_runtime.c        Scalar kernels using malloc/free
│   ├── cnn_models_c.h      Public declarations for all 10 run_*() functions
│   ├── cnn_models_c.c      Model forward passes (layer stacking + data flow)
│   ├── main.c              Demo driver: runs all models, prints checksums
│   └── Makefile            Builds deepbindi_c_demo; `make debug` enables output
│
└── static_port/            ── Static variant ───────────────────────────────────
    ├── deepbindi_config.h  Logging + fatal-error macros (embedded portability)
    ├── arena.h             Pool size defines + bump-allocator prototypes
    ├── arena.c             Global arrays g_weight_pool / g_act_arena + allocators
    ├── nn_runtime.h        Same API as c_port (tensor_create → act_alloc)
    ├── nn_runtime.c        Same kernels; tensor_free / layer_free are no-ops
    ├── cnn_models_c.h      Same as c_port (unchanged)
    ├── cnn_models_c.c      Same models; each run_*() starts with act_arena_reset()
    ├── main.c              Driver; calls arena_stats() after all models
    └── Makefile            Builds deepbindi_static_demo; `make debug` / `make verify`
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

| Issue | Status |
|-------|--------|
| `int` width | All shape fields are plain `int`. On 16-bit MCUs (AVR, MSP430) `int` is 16-bit; tensor sizes for the larger 2-D models exceed 32 767. Use a 32-bit toolchain or change shape fields to `int32_t`. |
| `float` vs `double` | All arithmetic is `float`. No implicit promotion to `double` in the compute kernels. |
| `memset` to zero for `float` | Relies on IEEE-754 (all-zero bits = 0.0f). Safe on all Cortex-M, RISC-V, and x86 targets in common use. |
| `%zu` format specifier | Not supported by newlib-nano. `arena_stats()` uses `(unsigned)` casts + `%u` instead. |
| `stdio.h` / `stdlib.h` | Only included (transitively via `deepbindi_config.h`) when `DEEPBINDI_ENABLE_LOGGING` is defined. |

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

1. Export PyTorch weights to a flat binary (e.g. `torch.save` + a custom
   extraction script, or ONNX export + `onnx` Python package).
2. Replace the `*_layer_create()` calls in `cnn_models_c.c` with a loader that
   fills pre-allocated `float` arrays from the binary file.
3. For the **static** variant, pre-populate `g_weight_pool[]` at link time using
   a generated C header (`weights_cnn_1d_v1.h`) with the trained values as a
   `static const float` array, then memcpy into the pool during initialisation.
4. Verify correctness by comparing `tensor_checksum` against a Python reference
   forward pass on the same input values.

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

**Context:** The paper presents a fear-recognition system based on physiological signals (BVP, SKT, GSR) from the WEMAC dataset, achieving 80% F1-score and 74% accuracy. The system was validated on an ultra-low-power ARM Cortex-M4 (16 mW @ 5 V, 496 ms per inference). The C port in this repository implements the same model architectures to support CGRA-accelerated deployment on similar extreme-edge targets.
