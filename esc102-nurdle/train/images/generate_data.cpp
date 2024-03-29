#include <stdio.h>
#include <cmath>
#include <vector>
#include <string>

using std::min;
using std::max;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define S 49
struct Tile {
    uint8_t data[3][S][S];
};
struct vec3 {
    float x[3];
    vec3(float v = 0.0) {
        x[0] = x[1] = x[2] = v;
    }
};

int W, H;
uint8_t *imgin;
uint8_t *imgout;

void getPixel(
    uint8_t *img, int n_channel,
    int x, int y, uint8_t *res
) {
    x = max(0, min(W-1, x));
    y = max(0, min(H-1, y));
    int i = y * W + x;
    uint8_t *p = &img[n_channel*i];
    for (int i = 0; i < n_channel; i++)
        res[i] = p[i];
}

void getPixel(
    uint8_t *img, int n_channel,
    float x, float y, uint8_t *out
) {
    x *= W, y *= H;
    int px = (int)floor(x);
    int py = (int)floor(y);
    float fx = x - px;
    float fy = y - py;
    uint8_t c00[4], c10[4], c01[4], c11[4];
    getPixel(img, n_channel, px, py, c00);
    getPixel(img, n_channel, px+1, py, c10);
    getPixel(img, n_channel, px, py+1, c01);
    getPixel(img, n_channel, px+1, py+1, c11);
    for (int i = 0; i < n_channel; i++) {
        float a = (1.0-fx)*c00[i] + fx*c10[i];
        float b = (1.0-fx)*c01[i] + fx*c11[i];
        float c = (1.0-fy)*a + fy*b;
        if (n_channel == 1)
            c = min(max(1.1*(c-128.)+128., 0.0), 255.0);
        out[i] = (uint8_t)(c+0.5);
    }
}


std::vector<Tile> tiles;
std::vector<uint8_t> values;
std::vector<float> weights;
std::vector<vec3> means, invstdevs;

void addTile(float cx, float cy, float rx, float ry, float angle) {
    Tile t;
    float ca = cos(angle), sa = sin(angle);

    // in: tile
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            float u = i * 2.0 / (S-1.0) - 1.0;
            float v = j * 2.0 / (S-1.0) - 1.0;
            float x = cx + rx * (ca*u+sa*v);
            float y = cy + ry * (ca*v-sa*u);
            uint8_t out[3];
            getPixel(imgin, 3, x, y, out);
            for (int _ = 0; _ < 3; _++)
                t.data[_][i][j] = out[_];
        }
    }
    tiles.push_back(t);

    // out: max of neighbors
    float sx = 0.01 * min(W, H) / (float)W;
    float sy = 0.01 * min(W, H) / (float)H;
    uint8_t out = 0;
    const int s = 8;
    for (int i = -s; i <= s; i++) {
        for (int j = -s; j <= s; j++) {
            float u = i / (float)s;
            float v = j / (float)s;
            float x = cx + sx * (ca*u+sa*v);
            float y = cy + sy * (ca*v-sa*u);
            uint8_t p;
            getPixel(imgout, 1, x, y, &p);
            out = max(p, out);
        }
    }
    values.push_back(out);

    // mean and variance
    float scx = min(W, H) / (float)W;
    float scy = min(W, H) / (float)H;
    const float n = 24.0;
    vec3 s1(0), s2(0);
    for (float i = 0.5; i < n; i++) {
        const float phi = 0.5*(1.+sqrt(5.));
        float u1 = fmod(i/phi, 1.0), u2 = i/n;
        float a = 2.0*3.1415923*u2;
        float r = 0.2*sqrt(-2.0*log(1.0-u1));
        uint8_t c[3];
        getPixel(imgin, 3, cx+r*scx*cos(a), cy+r*scy*sin(a), c);
        for (int _ = 0; _ < 3; _++) {
            float t = c[_] / 255.0;
            s1.x[_] += t, s2.x[_] += t * t;
        }
    }
    vec3 mean, invstdev;
    for (int _ = 0; _ < 3; _++) {
        mean.x[_] = s1.x[_] / n;
        float var = (s2.x[_]-s1.x[_]*s1.x[_]/n)/(n-1.);
        invstdev.x[_] = 1.0/sqrt(var+1e-2);
    }
    means.push_back(mean);
    invstdevs.push_back(invstdev);
}

#define N_TILES 5000
#define N_IMGS 21

float vanDerCorput(int n, int b) {
    float x = 0.0, e = 1.0 / b;
    while (n) {
        x += (n % b) * e;
        e /= b, n /= b;
    }
    return x;
}
float processImage() {
    // check nurdle size
    float tot = 0.0;
    for (int i = 0; i < W; i++)
        for (int j = 0; j < H; j++) {
            uint8_t c;
            getPixel(imgout, 1, i/(float)W, j/(float)H, &c);
            tot += (c/255.0);
        }
    float r = pow(tot / (W*H), 0.5);
    float sr = r==0.0 ? 1.0 : r/0.1;
    // printf("%f\n", r);

    float sx = min(W, H) / (float)W;
    float sy = min(W, H) / (float)H;
    tot = 0.0;
    for (int ti = 0; ti < N_TILES; ti++) {
        float s = vanDerCorput(ti, 7);
        s = sr*exp(log(0.01)+log(0.25/0.01)*pow(s,0.5));
        addTile(
            0.05+0.9*vanDerCorput(ti, 2),
            0.05+0.9*vanDerCorput(ti, 3),
            s * sx,
            s * sy,
            2.0*3.1415926*vanDerCorput(ti, 5)
        );
        tot += (float)values.back() / 255.0;
    }
    return tot / N_TILES;
}

int main() {
    tiles.reserve(N_TILES*N_IMGS);
    values.reserve(N_TILES*N_IMGS);
    weights.reserve(N_TILES*N_IMGS);
    means.reserve(N_TILES*N_IMGS);
    invstdevs.reserve(N_TILES*N_IMGS);
    for (int i = 0; i < N_IMGS; i++) {
        std::string filename("00");
        sprintf(&filename[0], "%02d", i%21);  // first 2 weight more
        std::string fin = "train/"+filename+".jpg";
        std::string fout = "train/"+filename+".png";
        imgin = stbi_load(&fin[0], &W, &H, nullptr, 3);
        imgout = stbi_load(&fout[0], &W, &H, nullptr, 1);
        float p = processImage();
        printf("%02d  %d %d  %f\n", i, W, H, p);
        // printf("%f,", i, W, H, p);
        float w = exp(-p/0.05);
        for (int i = 0; i < N_TILES; i++)
            weights.push_back(w);
        delete imgin; delete imgout;
    }
    FILE* fp = fopen("data_in.raw", "wb");
    for (Tile t : tiles)
        fwrite(t.data, 1, 3*S*S, fp);
    fclose(fp);
    fp = fopen("data_out.raw", "wb");
    fwrite(&values[0], 1, values.size(), fp);
    fclose(fp);
    fp = fopen("data_weight.raw", "wb");
    fwrite(&weights[0], 4, weights.size(), fp);
    fclose(fp);
    assert(sizeof(vec3) == 12);
    fp = fopen("data_mean.raw", "wb");
    fwrite(&means[0], 12, means.size(), fp);
    fclose(fp);
    fp = fopen("data_invstdev.raw", "wb");
    fwrite(&invstdevs[0], 12, invstdevs.size(), fp);
    fclose(fp);
    return 0;
}
