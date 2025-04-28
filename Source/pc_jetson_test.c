#include "lenet_quantized.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//Update your paths!
#define FILE_TEST_IMAGE		"X:\\Development\\LeNet5_c_cuda_quantized\\Library\\LeNet5\\LeNet-5\\t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"X:\\Development\\LeNet5_c_cuda_quantized\\Library\\LeNet5\\LeNet-5\\t10k-labels-idx1-ubyte"
#define LENET_FILE 		"X:\\Development\\LeNet5_c_cuda_quantized\\Output\\model.dat"
#define LENET_Q_FILE 	"X:\\Development\\LeNet5_c_cuda_quantized\\Output\\model_quantized.dat"
#define COUNT_TEST       10000

int read_data(image data[], uint8 label[], const int count, const char* data_file, const char* label_file) {
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image || !fp_label) return 1;
    fseek(fp_image, 16, SEEK_SET);
    fseek(fp_label, 8, SEEK_SET);
    fread(data, sizeof(image) * count, 1, fp_image);
    fread(label, sizeof(uint8) * count, 1, fp_label);
    fclose(fp_image);
    fclose(fp_label);
    return 0;
}

int load_model(LeNet5* lenet, const char* filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return 1;
    fread(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

int main() {
    image *test_data = (image*)malloc(sizeof(image) * COUNT_TEST);
    uint8 *test_label = (uint8*)malloc(sizeof(uint8) * COUNT_TEST);
    if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL)) {
        printf("Error reading MNIST test set.\n");
        return 1;
    }

    LeNet5* lenet = (LeNet5*)malloc(sizeof(LeNet5));
    if (load_model(lenet, LENET_FILE)) {
        printf("Error loading model.dat.\n");
        return 1;
    }

    clock_t start = clock();
    int correct = 0;
    for (int i = 0; i < COUNT_TEST; ++i) {
        uint8 pred = Predict(lenet, test_data[i], OUTPUT);
        if (pred == test_label[i]) correct++;
    }
    clock_t end = clock();

    printf("Accuracy: %d / %d\n", correct, COUNT_TEST);
    printf("Elapsed Time: %.2f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(test_data);
    free(test_label);
    free(lenet);
    return 0;
}