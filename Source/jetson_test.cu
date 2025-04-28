#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define LENGTH_KERNEL	5
#define LENGTH_FEATURE0	32
#define LENGTH_FEATURE1	(LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE2	(LENGTH_FEATURE1 >> 1)
#define LENGTH_FEATURE3	(LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE4	(LENGTH_FEATURE3 >> 1)
#define LENGTH_FEATURE5	(LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

#define INPUT	1
#define LAYER1	6
#define LAYER2	6
#define LAYER3	16
#define LAYER4	16
#define LAYER5	120
#define OUTPUT 10
#define PADDING 2
#define COUNT_TEST 10000

typedef unsigned char uint8;
typedef uint8 image[28][28];

struct LeNet5 {
    double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];

    double bias0_1[LAYER1];
    double bias2_3[LAYER3];
    double bias4_5[LAYER5];
    double bias5_6[OUTPUT];
};

struct Feature {
    double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
    double layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
    double layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
    double layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
    double layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
    double layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
    double output[OUTPUT];
};

__device__ double relu(double x) {
    return x > 0 ? x : 0;
}

__device__ void conv2d_valid_32_28(const double in[32][32], double out[28][28], const double kernel[5][5]) {
    for (int i = 0; i < 28; ++i)
        for (int j = 0; j < 28; ++j)
            for (int ki = 0; ki < 5; ++ki)
                for (int kj = 0; kj < 5; ++kj)
                    out[i][j] += in[i + ki][j + kj] * kernel[ki][kj];
}

__device__ void conv2d_valid_14_10(const double in[14][14], double out[10][10], const double kernel[5][5]) {
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
            for (int ki = 0; ki < 5; ++ki)
                for (int kj = 0; kj < 5; ++kj)
                    out[i][j] += in[i + ki][j + kj] * kernel[ki][kj];
}

__device__ void conv2d_valid_10_6(const double in[10][10], double out[6][6], const double kernel[5][5]) {
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            for (int ki = 0; ki < 5; ++ki)
                for (int kj = 0; kj < 5; ++kj)
                    out[i][j] += in[i + ki][j + kj] * kernel[ki][kj];
}

__device__ void conv2d_valid_5_1(const double in[5][5], double out[1][1], const double kernel[5][5]) {
    for (int ki = 0; ki < 5; ++ki)
        for (int kj = 0; kj < 5; ++kj)
            out[0][0] += in[ki][kj] * kernel[ki][kj];
}

__device__ void maxpool2x2(const double in[28][28], double out[14][14]) {
    for (int i = 0; i < 14; ++i)
        for (int j = 0; j < 14; ++j) {
            double maxval = in[i*2][j*2];
            for (int m = 0; m < 2; ++m)
                for (int n = 0; n < 2; ++n)
                    maxval = fmax(maxval, in[i*2 + m][j*2 + n]);
            out[i][j] = maxval;
        }
}

__device__ void maxpool2x2_10_5(const double in[10][10], double out[5][5]) {
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j) {
            double maxval = in[i*2][j*2];
            for (int m = 0; m < 2; ++m)
                for (int n = 0; n < 2; ++n)
                    maxval = fmax(maxval, in[i*2 + m][j*2 + n]);
            out[i][j] = maxval;
        }
}

__device__ void fc_forward(const double in[], double out[], const double weight[][OUTPUT], const double bias[], int in_size) {
    for (int i = 0; i < OUTPUT; ++i) {
        out[i] = bias[i];
        for (int j = 0; j < in_size; ++j)
            out[i] += in[j] * weight[j][i];
        out[i] = relu(out[i]);
    }
}

__device__ void flatten(const double in[LAYER5][1][1], double out[LAYER5]) {
    for (int i = 0; i < LAYER5; ++i)
        out[i] = in[i][0][0];
}

__device__ uint8 forward_cuda(const image input, LeNet5* lenet) {
    double input_pad[32][32] = {0};
    double mean = 0, std = 0;
    for (int i = 0; i < 28; ++i)
        for (int j = 0; j < 28; ++j) {
            mean += input[i][j];
            std += input[i][j] * input[i][j];
        }
    mean /= (28 * 28);
    std = sqrt(std / (28 * 28) - mean * mean);
    for (int i = 0; i < 28; ++i)
        for (int j = 0; j < 28; ++j)
            input_pad[i + PADDING][j + PADDING] = (input[i][j] - mean) / std;

    Feature f = {0};
    for (int i = 0; i < 32; ++i)
        for (int j = 0; j < 32; ++j)
            f.input[0][i][j] = input_pad[i][j];

    for (int o = 0; o < LAYER1; ++o) {
        conv2d_valid_32_28(f.input[0], f.layer1[o], lenet->weight0_1[0][o]);
        for (int i = 0; i < 28; ++i)
            for (int j = 0; j < 28; ++j)
                f.layer1[o][i][j] = relu(f.layer1[o][i][j] + lenet->bias0_1[o]);
        maxpool2x2(f.layer1[o], f.layer2[o]);
    }
    for (int o = 0; o < LAYER3; ++o) {
        for (int i = 0; i < LAYER2; ++i)
            conv2d_valid_14_10(f.layer2[i], f.layer3[o], lenet->weight2_3[i][o]);
        for (int i = 0; i < 10; ++i)
            for (int j = 0; j < 10; ++j)
                f.layer3[o][i][j] = relu(f.layer3[o][i][j] + lenet->bias2_3[o]);
        maxpool2x2_10_5(f.layer3[o], f.layer4[o]);
    }
    for (int o = 0; o < LAYER5; ++o) {
        for (int i = 0; i < LAYER4; ++i)
            conv2d_valid_5_1(f.layer4[i], f.layer5[o], lenet->weight4_5[i][o]);
        f.layer5[o][0][0] = relu(f.layer5[o][0][0] + lenet->bias4_5[o]);
    }
    double fc_in[LAYER5];
    flatten(f.layer5, fc_in);
    fc_forward(fc_in, f.output, lenet->weight5_6, lenet->bias5_6, LAYER5);

    uint8 pred = 0;
    double maxval = f.output[0];
    for (int i = 1; i < OUTPUT; ++i) {
        if (f.output[i] > maxval) {
            maxval = f.output[i];
            pred = i;
        }
    }
    return pred;
}

__global__ void predict_kernel(const image *inputs, LeNet5 *lenet, uint8 *results) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < COUNT_TEST)
        results[i] = forward_cuda(inputs[i], lenet);
}

//Update your paths!
int main() {
    FILE *fp = fopen("model.dat", "rb");
    if (!fp) { printf("Cannot open model.dat\n"); return 1; }
    LeNet5 *h_model = (LeNet5 *)malloc(sizeof(LeNet5));
    fread(h_model, sizeof(LeNet5), 1, fp);
    fclose(fp);

    FILE *fimg = fopen("t10k-images-idx3-ubyte", "rb");
    FILE *flbl = fopen("t10k-labels-idx1-ubyte", "rb");
    if (!fimg || !flbl) { printf("Cannot open MNIST test set\n"); return 1; }
    fseek(fimg, 16, SEEK_SET);
    fseek(flbl, 8, SEEK_SET);

    image *h_images = (image *)malloc(sizeof(image) * COUNT_TEST);
    uint8 *h_labels = (uint8 *)malloc(sizeof(uint8) * COUNT_TEST);
    fread(h_images, sizeof(image), COUNT_TEST, fimg);
    fread(h_labels, sizeof(uint8), COUNT_TEST, flbl);
    fclose(fimg);
    fclose(flbl);

    image *d_images;
    uint8 *d_results;
    cudaMalloc(&d_images, sizeof(image) * COUNT_TEST);
    cudaMalloc(&d_results, sizeof(uint8) * COUNT_TEST);

    cudaMemcpy(d_images, h_images, sizeof(image) * COUNT_TEST, cudaMemcpyHostToDevice);

    LeNet5 *d_model;
    cudaMalloc(&d_model, sizeof(LeNet5));
    cudaMemcpy(d_model, h_model, sizeof(LeNet5), cudaMemcpyHostToDevice);

    clock_t start = clock();
    predict_kernel<<<(COUNT_TEST + 255) / 256, 256>>>(d_images, d_model, d_results);
    cudaDeviceSynchronize();
    clock_t end = clock();

    uint8 *h_results = (uint8 *)malloc(sizeof(uint8) * COUNT_TEST);
    cudaMemcpy(h_results, d_results, sizeof(uint8) * COUNT_TEST, cudaMemcpyDeviceToHost);

    int correct = 0;
    for (int i = 0; i < COUNT_TEST; ++i)
        if (h_results[i] == h_labels[i]) correct++;

    printf("Accuracy: %d / %d\n", correct, COUNT_TEST);
    printf("Elapsed Time: %.2f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    cudaFree(d_model);
    cudaFree(d_images);
    cudaFree(d_results);
    free(h_model);
    free(h_images);
    free(h_labels);
    free(h_results);
    return 0;
}