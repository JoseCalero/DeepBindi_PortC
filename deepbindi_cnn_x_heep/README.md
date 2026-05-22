# deepbindi_cnn_x_heep

**CNN_1D_v2 (Model 4 / PYE) -- int32 C port for X-HEEP (RISC-V)**

A fully integer, heap-free, FPU-free inference engine for the DeepBindi PYE model,
designed for deployment on the [X-HEEP](https://github.com/esl-epfl/x-heep) RISC-V SoC
with optional STRELA CGRA acceleration.

---

## Model

**CNN_1D_v2** (also called PYE or Model 4 in the DeepBindi paper):

```
Input   : (1, 57, 1, 10)  -- 57 features x 10 time frames (WEMAC dataset)

Block 1 : Conv1d(57->32, k=5, pad=1) -> BN(32) -> ReLU -> MaxPool1d(2)
          out_w = (10 + 2 - 5)/1 + 1 = 8  -> MaxPool -> 4
          shape: (1, 32, 1, 4)

Block 2 : Conv1d(32->64, k=5, pad=1) -> BN(64) -> ReLU -> MaxPool1d(2)
          out_w = (4 + 2 - 5)/1 + 1 = 2  -> MaxPool -> 1
          shape: (1, 64, 1, 1)

Head    : Flatten -> Dense(64->1) -> threshold(0)
          output = 1 (FEAR) if pre-threshold value > 0, else 0 (NO_FEAR)
```

Trained on WEMAC (wemac_50Overlapping.csv), 10-fold CV:
- Best fold (0): test F1 = 0.792
- Mean across 10 folds: F1 = 0.745, Acc = 0.717

---

## Files

| File | Purpose |
|------|---------|
| `main.c` | Entry point: runs 2 test samples, prints output + cycle count |
| `nn_runtime.c/h` | Tensor, Conv2D, BatchNorm, MaxPool, Dense, activations (int32) |
| `cnn_models_c.c/h` | `run_cnn_1d_v2()` -- full forward pass |
| `arena.c/h` | Static bump allocators (weight pool + activation arena) |
| `deepbindi_config.h` | Pool sizes, logging macros, CSR stubs for PC build |
| `test_input.h` | Two WEMAC fold-0 test samples (NO_FEAR and FEAR) |
| `weights_cnn_1d_v2.h` | Trained weights as const int32_t arrays (generated) |
| `export_pytorch_weights.py` | Convert .pth checkpoint to `weights_cnn_1d_v2.h` |
| `extract_test_inputs.py` | Extract WEMAC samples and write `test_input.h` |
| `Makefile` | PC build (dummy and real-weights targets) |

Python training scripts (in `EPFL_STAY_LAURA/python_code/simulations/`):
- `train_cnn1d_v2_xheep.py` -- standalone PyTorch training, produces `.pth` checkpoints

---

## Quick start: PC build

### Prerequisites

- GCC (MSYS2 UCRT64 on Windows, or native gcc on Linux/macOS)
- Python 3.x with `torch`, `numpy` (for weight export and test-input extraction)
- Trained checkpoint (`type4_fold0.pth`) -- see Training section below

### Build dummy-weights binary (no .pth required)

```bash
make
make run
```

The dummy build uses seeded random weights for benchmarking execution flow
and cycle counting. Classification output is not meaningful.

### Build real-weights binary

1. Train the model (or use existing checkpoint):
   ```
   results/deepBindi/cnn1d_v2_xheep/type4_fold0.pth
   ```

2. Export weights to C header:
   ```bash
   python export_pytorch_weights.py
   # auto-detects most recent checkpoint; writes weights_cnn_1d_v2.h
   ```

3. Extract test inputs:
   ```bash
   python extract_test_inputs.py
   # writes test_input.h with two WEMAC fold-0 samples
   ```

4. Build and run:
   ```bash
   make run-real
   ```

**Expected output:**
```
DeepBindi CNN_1D_v2 on X-HEEP (int32)
--- Sample 0 (expected label=0 / NO_FEAR) ---
Output : 0 (NO_FEAR)
--- Sample 1 (expected label=1 / FEAR) ---
Output : 1 (FEAR)
-- Arena usage --
  Weight pool : 0 / 24000 words  (0 / 93 KB)
  Act arena   : 1211 / 2048 words  (4 / 8 KB)
  Tensor pool : 7 / 16 structs
```

### Windows / MSYS2 note

`make` cannot find gcc if the UCRT64 bin directory is not on PATH.
Use the explicit approach:

```bash
export PATH="/c/msys64/ucrt64/bin:$PATH"
gcc -Wall -Wextra -O2 -std=c99 -DTARGET_PC -c -o main.o main.c
# ... (see Makefile for all object files)
gcc -Wall -Wextra -O2 -std=c99 -DTARGET_PC -o deepbindi_cnn_pc \
    main.o nn_runtime.o cnn_models_c.o arena.o

# Real-weights variant:
gcc -Wall -Wextra -O2 -std=c99 -DTARGET_PC -DDEEPBINDI_REAL_WEIGHTS \
    -c -o main_real.o main.c
# ... (add -DDEEPBINDI_REAL_WEIGHTS to all object compilations)
gcc -Wall -Wextra -O2 -std=c99 -DTARGET_PC -DDEEPBINDI_REAL_WEIGHTS \
    -o deepbindi_cnn_pc_real \
    main_real.o nn_runtime_real.o cnn_models_c_real.o arena_real.o
./deepbindi_cnn_pc_real.exe
```

---

## Training

The model is trained with a standalone PyTorch script that avoids TensorFlow/Keras:

```bash
cd EPFL_STAY_LAURA/python_code/simulations/
python train_cnn1d_v2_xheep.py
```

Output directory: `results/deepBindi/cnn1d_v2_xheep/`
- `type4_fold0.pth` ... `type4_fold9.pth` -- per-fold checkpoints
- `best_model.pth` -- fold with highest test F1
- `fold_metrics.csv` -- per-fold summary
- `training_log.txt` -- epoch-level log

**Prerequisite** (Windows): `src/deep` must be a junction pointing to `src/CH07_DeepLearning`:
```cmd
cd EPFL_STAY_LAURA\python_code\src
mklink /j deep CH07_DeepLearning
```
The old `python-deep 0.10` package (if installed) must be removed:
```bash
pip uninstall deep -y
```

---

## Weight export

`export_pytorch_weights.py` performs:

1. Per-tensor symmetric int8 quantization of Conv/Dense weights:
   ```
   scale_f  = max(|w|) / 127
   w_int    = round(w / scale_f)     # in [-127, 127]
   ```

2. BatchNorm pre-folding to Q7 scale/offset (avoids per-inference sqrt):
   ```
   scale[c]  = round(gamma[c] / sqrt(var[c] + eps) * 128)   # Q7: 128 = 1.0
   offset[c] = round(beta[c]  - gamma[c] * mean[c] / sqrt(var[c] + eps))
   ```

Observed ranges for fold-0 checkpoint:
- BN1 scale: 245 -- 377
- BN2 scale: 184 -- 334

---

## Test input format

`extract_test_inputs.py` selects the most confidently classified NO_FEAR and FEAR
samples from fold 0's test set and writes them to `test_input.h`.

Input quantization:
```
x_int32 = round(x_float * 12).clip(-128, 127)
```

The scale factor 12 maps z-score normalized WEMAC features (~[-10, 10]) into int8
range (~[-120, 120]), consistent with the overflow analysis below.

Float-model confidence for the selected fold-0 samples:
- Sample 0 (NO_FEAR): sigmoid = 0.0313  (well below 0.5)
- Sample 1 (FEAR):    sigmoid = 0.9987  (well above 0.5)

After quantization the float model gives 0.0314 and 0.9987 respectively --
quantization error is negligible for these high-confidence samples.

---

## Integer arithmetic and overflow

All computations use `int32_t` throughout. Potential overflow points are mitigated:

### BatchNorm
Uses `int64_t` intermediate:
```c
y = (int32_t)(((int64_t)x * scale[c]) >> 7) + offset[c]
```

### Post-BN right-shifts (real-weights path only)

After BN1 (Q7 scale <= 377), activations can reach ~13.6M.
Conv2 (160 MACs x 127) would accumulate 276B -- well above INT32_MAX (2.1B).

Solution: `tensor_rshift_inplace()` after each BN+ReLU in the real-weights path:

```
After BN1 + ReLU:  >> 9   (13.6M >> 9 ~= 26.6K;  Conv2 max ~= 540M  OK)
After BN2 + ReLU:  >> 14  (1.41B >> 14 ~= 86K;    FC1  max ~= 699M  OK)
```

These shifts are only applied in the `DEEPBINDI_REAL_WEIGHTS` path.
The dummy path uses scale=128 (identity BN) and tiny weights -- no overflow.

---

## Memory footprint

| Region | Size | Usage |
|--------|------|-------|
| Weight pool | 24000 words (93 KB) | 19713 words in dummy build; 0 in real (ROM) |
| Activation arena | 2048 words (8 KB) | 1211 words peak |
| Tensor pool | 16 structs | 7 structs peak |
| `weights_cnn_1d_v2.h` | 19713 int32_t (77 KB) | Compiled into .rodata / flash |

---

## X-HEEP integration

Add this directory as an X-HEEP application and build via the CMake flow:

```cmake
target_compile_definitions(deepbindi PRIVATE TARGET_XHEEP DEEPBINDI_REAL_WEIGHTS)
```

Remove the `-DTARGET_PC` define; the CSR macros in `deepbindi_config.h` will
resolve to the real X-HEEP CSR access macros.

CGRA acceleration targets (see comments in `nn_runtime.c`):
- **PRIMARY**: `conv2d_forward()` -- innermost `(kh, kw)` MAC loops
- **SECONDARY**: `dense_forward()` -- dot-product over `in_features`
- **FUSIBLE**: `batchnorm_forward_inplace()` -- fuse with conv output stage

---

## Sigmoid replacement

The float sigmoid `1/(1+exp(-x))` is replaced by a hard threshold:
```c
output = (x > 0) ? 1 : 0
```
Valid for the single-output binary head of CNN_1D_v2 -- the sign of the
pre-sigmoid value determines the class. Avoids `expf()` entirely.
