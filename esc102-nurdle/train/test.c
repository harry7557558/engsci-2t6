#pragma GCC optimize "Ofast"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define byte unsigned char
#define max(x, y) ((x)>(y) ? (x) : (y))


float* load_weights(const char* path) {
    FILE* fp = fopen(path, "rb");
    fseek(fp, 0L, SEEK_END);
    int nw = ftell(fp) / sizeof(float);
    rewind(fp);
    float* w = malloc(sizeof(float) * nw);
    fread(w, sizeof(float), nw, fp);
    fclose(fp);
    printf("%d weights loaded from %s.\n", nw, path);
    return w;
}


void batchNorm(
    int n, float *x,
    float a, float b, float mu_, float var_
) {
#if 0
    // high variance, bad
    float s1 = 0.0, s2 = 0.0;
    for (int i = 0; i < n; i++) {
        s1 += x[i];
        s2 += x[i]*x[i];
    }
    float mu = s1 / (float)n;
    float var = (s2-s1*s1/(float)n)/(float)(n-0);
#else
    float mu = mu_, var = var_;  // simplify to linear??
#endif
    float invs = 1.0/sqrt(var+1e-5);
    for (int i = 0; i < n; i++) {
        x[i] = invs*(x[i]-mu) * a + b;
    }
}

void conv2d(
    int n, float *src, float *res,
    int ks, float *w,
    int st, int pd
) {
    int m = (n+2*pd-ks)/st+1;
    if (m & (m-1) != 0)
        printf("%d ", m);
    for (int ri = 0; ri < m; ri++) {
        for (int rj = 0; rj < m; rj++) {
            int i0 = st*ri-pd;
            int j0 = st*rj-pd;
            float s = 0.0;
            for (int i = 0; i < ks; i++) {
                for (int j = 0; j < ks; j++) {
                    int i1 = i0+i;
                    int j1 = j0+j;
                    if (i1 >= 0 && i1 < n && j1 >= 0 && j1 < n)
                        s += w[ks*i+j] * src[i1*n+j1];
                }
            }
            res[ri*m+rj] += s;
        }
    }
}

void leakyReLU(
    int n, float *x
) {
    for (int i = 0; i < n; i++) {
        x[i] = max(0.2*x[i], x[i]);
    }
}



const float bn00[3] = {1.2634873,0.53158915,0.8686848};
const float bn01[3] = {0.32725582,0.47777665,1.2413592};
const float bn10[4] = {0.9355589,0.6108142,1.4159594,1.0174707};
const float bn11[4] = {-0.22180158,0.1886153,-0.6549942,-0.2015215};
const float bn20[4] = {1.238508,1.0404031,0.9140648,0.96639305};
const float bn21[4] = {-0.6241729,-0.38913712,-1.0839738,-0.43033585};
const float bn30[2] = {1.6915355,1.7521297};
const float bn31[2] = {0.8405502,0.87476784};
const float bn02[3] = {0.66931975,0.66210276,0.62281305};
const float bn03[3] = {0.0064018695,0.0065998407,0.008052157};
const float bn12[4] = {0.42455563,-9.958383,4.1978683,-1.7027874};
const float bn13[4] = {5.340445,18.394554,7.318684,5.1953855};
const float bn22[4] = {-0.6425854,0.09796902,1.3378471,-0.43222108};
const float bn23[4] = {3.6591194,1.7221743,3.04153,4.916679};
const float bn32[2] = {-0.14095429,-0.2408477};
const float bn33[2] = {1.8572435,3.5620577};
float *w00, *w01, *w02, *w03;



#define S 31

/*
Model(
  (main): Sequential(
    (0): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Conv2d(3, 4, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
    (3): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace=True)
    (5): Conv2d(4, 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (6): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.2, inplace=True)
    (8): Conv2d(4, 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (9): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): LeakyReLU(negative_slope=0.2, inplace=True)
    (11): Conv2d(2, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (12): Sigmoid()
  )
)
*/
float calcPixel(float *x) {
    for (int i = 0; i < 3; i++)
        batchNorm(S*S, &x[S*S*i], bn00[i], bn01[i], bn02[i], bn03[i]);
    leakyReLU(3*S*S, x);
    // 3 x 31 x 31
    static float x1[4][16*16];
    memset(x1, 0, sizeof(x1));
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++)
            conv2d(31, &x[S*S*j], x1[i], 5, &w00[(3*i+j)*25], 2, 2);
        batchNorm(16*16, x1[i], bn10[i], bn11[i], bn12[i], bn13[i]);
        leakyReLU(16*16, x1[i]);
    }
    // 4 x 16 x 16
    static float x2[4][8*8];
    memset(x2, 0, sizeof(x2));
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)
            conv2d(16, x1[j], x2[i], 4, &w01[(4*i+j)*16], 2, 1);
        batchNorm(8*8, x2[i], bn20[i], bn21[i], bn22[i], bn23[i]);
        leakyReLU(8*8, x2[i]);
    }
    // 4 x 8 x 8
    static float x3[2][4*4];
    memset(x3, 0, sizeof(x3));
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++)
            conv2d(8, x2[j], x3[i], 4, &w02[(4*i+j)*16], 2, 1);
        batchNorm(4*4, x3[i], bn30[i], bn31[i], bn32[i], bn33[i]);
        leakyReLU(4*4, x3[i]);
    }
    // 2 x 4 x 4
    float x4 = 0.0;
    for (int j = 0; j < 2; j++)
        conv2d(4, x3[j], &x4, 4, &w03[j*16], 1, 0);
    // 1
    return 1.0/(1.0+exp(-x4));
}


int main() {

    // load weights
    w00 = load_weights("weights/w00_4_3_5_5.bin");
    w01 = load_weights("weights/w01_4_4_4_4.bin");
    w02 = load_weights("weights/w02_2_4_4_4.bin");
    w03 = load_weights("weights/w03_1_2_4_4.bin");
    printf("All weights loaded.\n");

    int w, h;
    byte* src = stbi_load("nurdles_test.jpg", &w, &h, NULL, 3);
    if (!src) return 0 * printf("Error\n");

    const int skip = 4;
    int wo = w/skip, ho = h/skip;
    byte* res = malloc(3*wo*ho);
    memset(res, 0, 3*wo*ho);

    float block[3*S*S];
    for (int j0 = 0; j0 < h-S; j0+=skip) {
        for (int i0 = 0; i0 < w-S; i0+=skip) {
            for (int j=0; j<S; j++) {
                for (int i=0; i<S; i++) {
                    byte *u = &src[3*((j0+j)*w+(i0+i))];
                    for (int k=0; k<3; k++) {
                        block[k*S*S+j*S+i] = (float)u[k] / 255.0;
                        // block[k*S*S+j*S+i] = (float)u[k];
                    }
                }
            }
            float v = calcPixel(block);
            // float v = block[0];
            byte *r = &res[3*((j0+S/2)/skip*wo+(i0+S/2)/skip)];
            for (int k = 0; k < 3; k++)
                r[k] = (byte)(255*v);
        }
    }

    stbi_write_png("nurdles_test_out.png",
        wo, ho,
        3, res, 3 * wo);

    free(src); free(res);
    free(w00); free(w01); free(w02); free(w03);
    return 0;
}

