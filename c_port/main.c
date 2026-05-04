/**
 * main.c  –  Demo driver: run all CNN models with dummy inputs and print results.
 *
 * Compile and run:
 *   make && ./deepbindi_c_demo
 *
 * Each model runner allocates dummy inputs (deterministic pseudo-random values),
 * builds all layer weights in-place, performs one complete forward pass, prints
 * the output tensor shape + a checksum, and frees all memory.
 *
 * Expected output (one line per model):
 *   <name> | shape=(N,C,H,W) | checksum=<float> | values=[...]
 *
 * No real trained weights are loaded – the sole purpose here is to exercise
 * every layer in the compute graph so that CGRA implementers can profile
 * and validate their hardware implementations against a known reference.
 */

#include "deepbindi_config.h"  /* logging + fatal error macros – include first */
#include "cnn_models_c.h"

typedef Tensor *(*ModelRunner)(void);

typedef struct {
    const char *name;
    ModelRunner runner;
} ModelEntry;

int main(void) {
    /* Register all models defined in cnn_models.py (PyTorch + Keras variants) */
    ModelEntry models[] = {
        {"CNN_2D_v1", run_cnn_2d_v1},
        {"CNN_2D_v2", run_cnn_2d_v2},
        {"CNN_2D_v3", run_cnn_2d_v3},
        {"CNN_1D_v1", run_cnn_1d_v1},
        {"CNN_1D_v2", run_cnn_1d_v2},
        {"CNN_1D_v3", run_cnn_1d_v3},
        {"MobileNetV3Custom", run_mobilenet_v3_custom},
        {"CNN_1d_tensorflow_sigmoid", run_cnn_1d_tensorflow_sigmoid},
        {"CNN_1d_tensorflow_softmax", run_cnn_1d_tensorflow_softmax},
        {"CNN_2d_tensorflow_softmax", run_cnn_2d_tensorflow_softmax}
    };
    int count = (int)(sizeof(models) / sizeof(models[0]));

    for (int i = 0; i < count; ++i) {
        Tensor *output = models[i].runner();
        /* Print shape + up to 8 output values so CGRA outputs can be diff'd */
        tensor_print_values(models[i].name, output, 8);
        tensor_free(output);
    }

    return 0;
}
