#version 300 es
precision highp float;

out vec4 fragColor;

uniform int ZERO;

uniform vec2 iResolution;
uniform sampler2D iSampler;

uniform sampler2D iW;  // weights
uniform int iKs;  // kernel size, odd
uniform int nIn, nOut;  // in/out channels, between 1 and 4
uniform vec4 bnMu, bnVar, bnA, bnB;  // batch norm, fixed

float getW_(int i) {
    int w = textureSize(iW, 0).x;
    vec4 r = texelFetch(iW, ivec2((i/4)%w, (i/4)/w), 0);
    return i%4==0? r.x : i%4==1 ? r.y : i%4==2 ? r.z : r.w;
}
float getW(int oi, int ii, int i, int j) {
    return getW_((oi*nIn+ii)*iKs*iKs + i*iKs+j);
}

void main() {
    float v[4];
    for (int oi = 0; oi < nOut; oi++) {
        v[oi] = 0.0;
        for (int i = 0; i < iKs; i++)
            for (int j = 0; j < iKs; j++) {
                vec4 c = texelFetch(iSampler,
                    ivec2(gl_FragCoord.xy) + ivec2(i, j) - iKs/2,
                    0);
                v[oi] += getW(oi, 0, i, j) * c.x;
                if (nIn > 1) v[oi] += getW(oi, 1, i, j) * c.y;
                if (nIn > 2) v[oi] += getW(oi, 2, i, j) * c.z;
                if (nIn > 3) v[oi] += getW(oi, 3, i, j) * c.w;
            }
    }
    vec4 s = vec4(v[0], v[1], v[2], v[3]);
    if (nOut == 1) {
        float o = 1.0 / (1.0+exp(-s.x));
        s = vec4(o, o, o, 1);
    }
    else {
        s = (s-bnMu)/sqrt(bnVar+1e-5) * bnA + bnB;
        s = max(s, 0.2*s);
    }
    fragColor = s;
}
