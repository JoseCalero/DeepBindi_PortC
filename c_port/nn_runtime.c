/**
 * nn_runtime.c  –  Reference C implementation of inference primitives.
 *
 * All operations are written as straightforward scalar loops with no external
 * dependencies (only libc math).  This file is the CGRA starting point: each
 * major function is annotated with notes on the loop structure and the
 * multiply-accumulate (MAC) intensity that CGRA mapping should exploit.
 *
 * Coding conventions:
 *   • Tensors are NCHW, heap-allocated floats.
 *   • Every forward function allocates a NEW output tensor; the caller is
 *     responsible for calling tensor_free() on it.
 *   • In-place functions (batchnorm, activations) modify the input tensor.
 *   • No global state; all functions are re-entrant.
 */
#include "deepbindi_config.h"  /* logging + fatal error macros – include first */
#include "nn_runtime.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Internal helpers ───────────────────────────────────────────────────── */

/* Row-major NCHW index into a flat float array. */
static int tensor_index(const Tensor *tensor, int n, int c, int h, int w) {
    return (((n * tensor->c + c) * tensor->h + h) * tensor->w + w);
}

/* Deterministic pseudo-random value for dummy weight / input generation.
 * Replace this with real weight loading (e.g. from a .bin file) for
 * deployment with actual trained parameters. */
static float seeded_value(int index, int seed, float scale) {
    int raw = ((index + 1) * (seed + 3) * 17) % 257;
    return scale * (((float)raw / 128.0f) - 1.0f);
}

Tensor *tensor_create(int n, int c, int h, int w) {
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    int count = n * c * h * w;
    tensor->n = n;
    tensor->c = c;
    tensor->h = h;
    tensor->w = w;
    tensor->data = (float *)calloc((size_t)count, sizeof(float));
    return tensor;
}

Tensor *tensor_clone(const Tensor *src) {
    Tensor *dst = tensor_create(src->n, src->c, src->h, src->w);
    memcpy(dst->data, src->data, (size_t)tensor_numel(src) * sizeof(float));
    return dst;
}

void tensor_free(Tensor *tensor) {
    if (tensor == NULL) {
        return;
    }
    free(tensor->data);
    free(tensor);
}

int tensor_numel(const Tensor *tensor) {
    return tensor->n * tensor->c * tensor->h * tensor->w;
}

void tensor_fill_dummy(Tensor *tensor, float scale, int seed) {
    int total = tensor_numel(tensor);
    for (int i = 0; i < total; ++i) {
        tensor->data[i] = seeded_value(i, seed, scale);
    }
}

float tensor_checksum(const Tensor *tensor) {
    float acc = 0.0f;
    int total = tensor_numel(tensor);
    for (int i = 0; i < total; ++i) {
        acc += tensor->data[i] * (float)((i % 7) + 1);
    }
    return acc;
}

void tensor_print_values(const char *name, const Tensor *tensor, int max_values) {
#ifdef DEEPBINDI_ENABLE_LOGGING
    int total = tensor_numel(tensor);
    int limit = total < max_values ? total : max_values;
    DEEPBINDI_PRINTF("%s | shape=(%d,%d,%d,%d) | checksum=%0.6f | values=[",
           name, tensor->n, tensor->c, tensor->h, tensor->w, tensor_checksum(tensor));
    for (int i = 0; i < limit; ++i) {
        DEEPBINDI_PRINTF("%s%0.6f", i == 0 ? "" : ", ", tensor->data[i]);
    }
    if (limit < total) {
        DEEPBINDI_PRINTF(", ...");
    }
    DEEPBINDI_PRINTF("]\n");
#else
    (void)name; (void)tensor; (void)max_values;
#endif
}

Conv2DLayer conv2d_layer_create(
    int in_channels,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int groups,
    int seed
) {
    Conv2DLayer layer;
    int in_per_group = in_channels / groups;
    int weight_count = out_channels * in_per_group * kernel_h * kernel_w;
    layer.in_channels = in_channels;
    layer.out_channels = out_channels;
    layer.kernel_h = kernel_h;
    layer.kernel_w = kernel_w;
    layer.stride_h = stride_h;
    layer.stride_w = stride_w;
    layer.pad_h = pad_h;
    layer.pad_w = pad_w;
    layer.groups = groups;
    layer.weights = (float *)malloc((size_t)weight_count * sizeof(float));
    layer.bias = (float *)malloc((size_t)out_channels * sizeof(float));

    for (int i = 0; i < weight_count; ++i) {
        layer.weights[i] = seeded_value(i, seed, 0.04f);
    }
    for (int i = 0; i < out_channels; ++i) {
        layer.bias[i] = seeded_value(i, seed + 11, 0.01f);
    }
    return layer;
}

void conv2d_layer_free(Conv2DLayer *layer) {
    free(layer->weights);
    free(layer->bias);
    layer->weights = NULL;
    layer->bias = NULL;
}

BatchNormLayer batchnorm_layer_create(int num_features, float eps, int seed) {
    BatchNormLayer layer;
    layer.num_features = num_features;
    layer.eps = eps;
    layer.gamma = (float *)malloc((size_t)num_features * sizeof(float));
    layer.beta = (float *)malloc((size_t)num_features * sizeof(float));
    layer.mean = (float *)malloc((size_t)num_features * sizeof(float));
    layer.var = (float *)malloc((size_t)num_features * sizeof(float));

    for (int i = 0; i < num_features; ++i) {
        layer.gamma[i] = 1.0f + fabsf(seeded_value(i, seed, 0.08f));
        layer.beta[i] = seeded_value(i, seed + 3, 0.02f);
        layer.mean[i] = seeded_value(i, seed + 7, 0.03f);
        layer.var[i] = 1.0f + fabsf(seeded_value(i, seed + 13, 0.05f));
    }
    return layer;
}

void batchnorm_layer_free(BatchNormLayer *layer) {
    free(layer->gamma);
    free(layer->beta);
    free(layer->mean);
    free(layer->var);
    layer->gamma = NULL;
    layer->beta = NULL;
    layer->mean = NULL;
    layer->var = NULL;
}

DenseLayer dense_layer_create(int in_features, int out_features, int seed) {
    DenseLayer layer;
    int weight_count = in_features * out_features;
    layer.in_features = in_features;
    layer.out_features = out_features;
    layer.weights = (float *)malloc((size_t)weight_count * sizeof(float));
    layer.bias = (float *)malloc((size_t)out_features * sizeof(float));

    for (int i = 0; i < weight_count; ++i) {
        layer.weights[i] = seeded_value(i, seed, 0.03f);
    }
    for (int i = 0; i < out_features; ++i) {
        layer.bias[i] = seeded_value(i, seed + 19, 0.02f);
    }
    return layer;
}

void dense_layer_free(DenseLayer *layer) {
    free(layer->weights);
    free(layer->bias);
    layer->weights = NULL;
    layer->bias = NULL;
}

/*
 * ═══════════════════════════════════════════════════════════════════════════
 * conv2d_forward  –  PRIMARY CGRA KERNEL
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * 2-D convolution (also 1-D when kernel_h == 1) with zero-padding and
 * optional grouped / depthwise convolution (groups > 1).
 *
 * Loop nest (7 levels):
 *   for n   in [0, batch)            – embarrassingly parallel per sample
 *     for oc  in [0, out_channels)   – output feature maps
 *       for oh  in [0, out_h)        – output rows
 *         for ow  in [0, out_w)      – output columns     <– spatial tile
 *           sum = bias[oc]
 *           for icg in [0, in_per_group) – input channels
 *             for kh  in [0, kernel_h)   – kernel row
 *               for kw  in [0, kernel_w) – kernel col
 *                 sum += input[n,ic,oh*s+kh,ow*s+kw] * weight[oc,icg,kh,kw]
 *           output[n,oc,oh,ow] = sum
 *
 * CGRA mapping hints
 * ──────────────────
 * • The innermost two loops (kh, kw) over the kernel window are the natural
 *   MAC chain: each iteration is one fused multiply-add with no loop-carried
 *   dependency across iterations.  Map these onto the CGRA functional units.
 *
 * • The (oh, ow) loops tile the spatial output; these can be distributed
 *   across CGRA rows/columns with no data dependency between tiles.
 *
 * • For depthwise convolutions (groups == in_channels == out_channels) the
 *   input-channel loop disappears (in_per_group == 1), reducing the kernel
 *   to a single-channel sliding window – simpler CGRA schedule.
 *
 * • Batch-norm and ReLU are typically fused directly after the accumulate
 *   before writing the output buffer (see apply_conv_bn_act in cnn_models_c.c).
 *
 * Weight memory layout:  weights[oc][icg][kh][kw]  (same as PyTorch)
 */
Tensor *conv2d_forward(const Tensor *input, const Conv2DLayer *layer) {
    int out_h = (input->h + 2 * layer->pad_h - layer->kernel_h) / layer->stride_h + 1;
    int out_w = (input->w + 2 * layer->pad_w - layer->kernel_w) / layer->stride_w + 1;
    int in_per_group = layer->in_channels / layer->groups;
    int out_per_group = layer->out_channels / layer->groups;
    Tensor *output = tensor_create(input->n, layer->out_channels, out_h, out_w);

    for (int n = 0; n < input->n; ++n) {
        for (int oc = 0; oc < layer->out_channels; ++oc) {
            int group = oc / out_per_group;
            int in_start = group * in_per_group; /* first input channel in this group */
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = layer->bias[oc];
                    for (int icg = 0; icg < in_per_group; ++icg) {
                        int ic = in_start + icg;
                        for (int kh = 0; kh < layer->kernel_h; ++kh) {
                            for (int kw = 0; kw < layer->kernel_w; ++kw) {
                                /* Translate output position + kernel offset to input position */
                                int ih = oh * layer->stride_h + kh - layer->pad_h;
                                int iw = ow * layer->stride_w + kw - layer->pad_w;
                                int weight_idx;
                                /* Zero-padding: skip out-of-bounds input positions */
                                if (ih < 0 || ih >= input->h || iw < 0 || iw >= input->w) {
                                    continue;
                                }
                                /* weights layout: [oc * in_per_group + icg][kh][kw] */
                                weight_idx =
                                    ((((oc * in_per_group) + icg) * layer->kernel_h + kh) * layer->kernel_w) + kw;
                                /* *** MAC: single multiply-accumulate – CGRA FU target *** */
                                sum += input->data[tensor_index(input, n, ic, ih, iw)] * layer->weights[weight_idx];
                            }
                        }
                    }
                    output->data[tensor_index(output, n, oc, oh, ow)] = sum;
                }
            }
        }
    }
    return output;
}

/*
 * batchnorm_forward_inplace
 *
 * Inference-mode batch normalisation (no running-statistics update).
 * For each channel c:
 *   y = gamma[c] * (x - mean[c]) / sqrt(var[c] + eps) + beta[c]
 *
 * This is a simple element-wise scale + shift per channel.  On a CGRA it
 * can be fused with the immediately preceding conv2d output stage to avoid
 * an extra round-trip through memory (compute gamma*inv_std once per channel,
 * then apply it together with bias[oc] inside the conv accumulator).
 */
void batchnorm_forward_inplace(Tensor *input, const BatchNormLayer *layer) {
    for (int n = 0; n < input->n; ++n) {
        for (int c = 0; c < input->c; ++c) {
            float gamma = layer->gamma[c];
            float beta = layer->beta[c];
            float mean = layer->mean[c];
            /* Pre-compute 1/sqrt(var+eps) once per channel to save divisions */
            float inv_std = 1.0f / sqrtf(layer->var[c] + layer->eps);
            for (int h = 0; h < input->h; ++h) {
                for (int w = 0; w < input->w; ++w) {
                    int idx = tensor_index(input, n, c, h, w);
                    input->data[idx] = gamma * (input->data[idx] - mean) * inv_std + beta;
                }
            }
        }
    }
}

/*
 * maxpool2d_forward
 *
 * Sliding-window maximum reduction.  Comparison-tree reduction over a small
 * (kernel_h × kernel_w) window – easy to implement on a CGRA reduction tree.
 */
Tensor *maxpool2d_forward(const Tensor *input, int kernel_h, int kernel_w, int stride_h, int stride_w) {
    int out_h = (input->h - kernel_h) / stride_h + 1;
    int out_w = (input->w - kernel_w) / stride_w + 1;
    Tensor *output = tensor_create(input->n, input->c, out_h, out_w);

    for (int n = 0; n < input->n; ++n) {
        for (int c = 0; c < input->c; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_value = -INFINITY;
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = oh * stride_h + kh;
                            int iw = ow * stride_w + kw;
                            float value = input->data[tensor_index(input, n, c, ih, iw)];
                            if (value > max_value) {
                                max_value = value;
                            }
                        }
                    }
                    output->data[tensor_index(output, n, c, oh, ow)] = max_value;
                }
            }
        }
    }
    return output;
}

/*
 * adaptive_avg_pool2d_forward
 *
 * Global average pooling (out_h == out_w == 1) is the most common use case,
 * reducing each feature map to a single scalar.  Used at the end of MobileNetV3
 * to collapse the spatial dimensions before the classifier head.
 */
Tensor *adaptive_avg_pool2d_forward(const Tensor *input, int out_h, int out_w) {
    Tensor *output = tensor_create(input->n, input->c, out_h, out_w);

    for (int n = 0; n < input->n; ++n) {
        for (int c = 0; c < input->c; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                int h_start = (oh * input->h) / out_h;
                int h_end = ((oh + 1) * input->h + out_h - 1) / out_h;
                for (int ow = 0; ow < out_w; ++ow) {
                    int w_start = (ow * input->w) / out_w;
                    int w_end = ((ow + 1) * input->w + out_w - 1) / out_w;
                    float sum = 0.0f;
                    int count = 0;
                    for (int ih = h_start; ih < h_end; ++ih) {
                        for (int iw = w_start; iw < w_end; ++iw) {
                            sum += input->data[tensor_index(input, n, c, ih, iw)];
                            ++count;
                        }
                    }
                    output->data[tensor_index(output, n, c, oh, ow)] = sum / (float)count;
                }
            }
        }
    }
    return output;
}

/*
 * flatten_forward
 *
 * Reshape (N, C, H, W) -> (N, C*H*W, 1, 1) via a shallow copy.
 * No arithmetic; pure memory reorganisation.
 */
Tensor *flatten_forward(const Tensor *input) {
    Tensor *output = tensor_create(input->n, tensor_numel(input) / input->n, 1, 1);
    memcpy(output->data, input->data, (size_t)tensor_numel(input) * sizeof(float));
    return output;
}

/*
 * dense_forward  –  SECONDARY CGRA KERNEL
 *
 * Fully-connected layer: y[out] = bias[out] + Σ_in (x[in] * W[out][in])
 *
 * This is a matrix-vector (or matrix-matrix for batch > 1) multiply.
 * On a CGRA it maps directly onto a MAC array: for each output neuron,
 * schedule a dot-product over in_features elements across the functional units.
 *
 * The weight matrix is read sequentially row-by-row (one row per output
 * neuron), which is cache-friendly for the weight buffer.
 */
Tensor *dense_forward(const Tensor *input, const DenseLayer *layer) {
    int features = tensor_numel(input) / input->n;
    Tensor *output = tensor_create(input->n, layer->out_features, 1, 1);

    if (features != layer->in_features) {
        DEEPBINDI_LOG_ERROR("dense_forward: expected %d features but got %d\n", layer->in_features, features);
        DEEPBINDI_FATAL("dense_forward shape mismatch");
    }

    for (int n = 0; n < input->n; ++n) {
        const float *src = input->data + n * layer->in_features;
        for (int out = 0; out < layer->out_features; ++out) {
            float sum = layer->bias[out];
            for (int in = 0; in < layer->in_features; ++in) {
                sum += src[in] * layer->weights[out * layer->in_features + in];
            }
            output->data[n * layer->out_features + out] = sum;
        }
    }
    return output;
}

/*
 * concat_height
 *
 * Concatenates two NCHW tensors along the H axis (like torch.cat(dim=2)).
 * Used in CNN_2D_v2 and CNN_2D_v3 to merge parallel conv branches that
 * operate at different feature-correlation scales before the second conv block.
 * This is a pure data-movement operation (no arithmetic).
 */
Tensor *concat_height(const Tensor *a, const Tensor *b) {
    Tensor *output;
    if (a->n != b->n || a->c != b->c || a->w != b->w) {
        DEEPBINDI_LOG_ERROR("concat_height: incompatible tensor shapes\n");
        DEEPBINDI_FATAL("concat_height shape mismatch");
    }

    output = tensor_create(a->n, a->c, a->h + b->h, a->w);
    for (int n = 0; n < output->n; ++n) {
        for (int c = 0; c < output->c; ++c) {
            for (int h = 0; h < a->h; ++h) {
                for (int w = 0; w < output->w; ++w) {
                    output->data[tensor_index(output, n, c, h, w)] = a->data[tensor_index(a, n, c, h, w)];
                }
            }
            for (int h = 0; h < b->h; ++h) {
                for (int w = 0; w < output->w; ++w) {
                    output->data[tensor_index(output, n, c, a->h + h, w)] = b->data[tensor_index(b, n, c, h, w)];
                }
            }
        }
    }
    return output;
}

/*
 * add_forward  –  Element-wise addition (residual connection)
 *
 * Used in MobileNetV3 bottleneck blocks when stride == 1 and
 * input_channels == output_channels.  The residual shortcut simply adds the
 * block input to its output element-by-element.  Trivially parallelisable.
 */
Tensor *add_forward(const Tensor *a, const Tensor *b) {
    int total;
    Tensor *output;
    if (a->n != b->n || a->c != b->c || a->h != b->h || a->w != b->w) {
        DEEPBINDI_LOG_ERROR("add_forward: incompatible tensor shapes\n");
        DEEPBINDI_FATAL("add_forward shape mismatch");
    }
    output = tensor_create(a->n, a->c, a->h, a->w);
    total = tensor_numel(a);
    for (int i = 0; i < total; ++i) {
        output->data[i] = a->data[i] + b->data[i];
    }
    return output;
}

/*
 * channel_scale_forward  –  SE channel re-weighting
 *
 * Part of the Squeeze-and-Excitation (SE) block in MobileNetV3:
 * after computing per-channel attention weights (shape N×C×1×1), multiply
 * each spatial position of every channel by its corresponding scalar.
 * Equivalent to: output[n,c,h,w] = input[n,c,h,w] * scale[n,c,0,0]
 */
Tensor *channel_scale_forward(const Tensor *input, const Tensor *scale) {
    Tensor *output;
    if (scale->n != input->n || scale->c != input->c || scale->h != 1 || scale->w != 1) {
        DEEPBINDI_LOG_ERROR("channel_scale_forward: expected scale shape (n,c,1,1)\n");
        DEEPBINDI_FATAL("channel_scale_forward shape mismatch");
    }
    output = tensor_create(input->n, input->c, input->h, input->w);
    for (int n = 0; n < input->n; ++n) {
        for (int c = 0; c < input->c; ++c) {
            float factor = scale->data[tensor_index(scale, n, c, 0, 0)];
            for (int h = 0; h < input->h; ++h) {
                for (int w = 0; w < input->w; ++w) {
                    int idx = tensor_index(input, n, c, h, w);
                    output->data[idx] = input->data[idx] * factor;
                }
            }
        }
    }
    return output;
}

void relu_inplace(Tensor *input) {
    int total = tensor_numel(input);
    for (int i = 0; i < total; ++i) {
        if (input->data[i] < 0.0f) {
            input->data[i] = 0.0f;
        }
    }
}

void sigmoid_inplace(Tensor *input) {
    int total = tensor_numel(input);
    for (int i = 0; i < total; ++i) {
        input->data[i] = 1.0f / (1.0f + expf(-input->data[i]));
    }
}

void hardsigmoid_inplace(Tensor *input) {
    int total = tensor_numel(input);
    for (int i = 0; i < total; ++i) {
        float value = (input->data[i] + 3.0f) / 6.0f;
        if (value < 0.0f) {
            value = 0.0f;
        }
        if (value > 1.0f) {
            value = 1.0f;
        }
        input->data[i] = value;
    }
}

void hardswish_inplace(Tensor *input) {
    int total = tensor_numel(input);
    for (int i = 0; i < total; ++i) {
        float gate = (input->data[i] + 3.0f) / 6.0f;
        if (gate < 0.0f) {
            gate = 0.0f;
        }
        if (gate > 1.0f) {
            gate = 1.0f;
        }
        input->data[i] *= gate;
    }
}

void softmax_inplace(Tensor *input) {
    int total = tensor_numel(input);
    float max_value = input->data[0];
    float sum = 0.0f;
    for (int i = 1; i < total; ++i) {
        if (input->data[i] > max_value) {
            max_value = input->data[i];
        }
    }
    for (int i = 0; i < total; ++i) {
        input->data[i] = expf(input->data[i] - max_value);
        sum += input->data[i];
    }
    for (int i = 0; i < total; ++i) {
        input->data[i] /= sum;
    }
}
