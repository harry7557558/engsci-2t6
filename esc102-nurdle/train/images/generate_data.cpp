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
        out[i] = (uint8_t)(c+0.5);
    }
}


std::vector<Tile> tiles;
std::vector<uint8_t> values;

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
}

#define N_TILES 5000
#define N_IMGS 23

float vanDerCorput(int n, int b) {
    float x = 0.0, e = 1.0 / b;
    while (n) {
        x += (n % b) * e;
        e /= b, n /= b;
    }
    return x;
}
void processImage() {
    float sx = min(W, H) / (float)W;
    float sy = min(W, H) / (float)H;
    for (int ti = 0; ti < N_TILES; ti++) {
        float s = vanDerCorput(ti, 7);
        s = exp(log(0.01)+log(0.25/0.01)*pow(s,0.5));
        addTile(
            0.025+0.95*vanDerCorput(ti, 2),
            0.025+0.95*vanDerCorput(ti, 3),
            s * sx,
            s * sy,
            2.0*3.1415926*vanDerCorput(ti, 5)
        );
    }
}

int main() {
    tiles.reserve(N_TILES*N_IMGS);
    values.reserve(N_TILES*N_IMGS);
    for (int i = 0; i < N_IMGS; i++) {
        std::string filename("00");
        sprintf(&filename[0], "%02d", i%21);  // first 2 weight more
        std::string fin = "train/"+filename+".jpg";
        std::string fout = "train/"+filename+".png";
        imgin = stbi_load(&fin[0], &W, &H, nullptr, 3);
        imgout = stbi_load(&fout[0], &W, &H, nullptr, 1);
        printf("%02d  %d %d\n", i, W, H);
        processImage();
        delete imgin; delete imgout;
    }
    FILE* fp = fopen("data_in.raw", "wb");
    for (Tile t : tiles)
        fwrite(t.data, 1, 3*S*S, fp);
    fclose(fp);
    fp = fopen("data_out.raw", "wb");
    fwrite(&values[0], 1, values.size(), fp);
    fclose(fp);
    return 0;
}
