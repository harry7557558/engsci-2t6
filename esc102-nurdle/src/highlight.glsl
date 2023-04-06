#version 300 es
precision highp float;

out vec4 fragColor;

uniform vec2 iResolution;

uniform sampler2D iSampler;  // image
uniform sampler2D nSampler;  // nurdle

float getN(vec2 xy) {
    vec2 coord = xy * iResolution.xy - 0.5;
    vec2 f = fract(coord);
    ivec2 p = ivec2(floor(coord));
    float c00 = texelFetch(nSampler, p+ivec2(0,0), 0).x;
    float c10 = texelFetch(nSampler, p+ivec2(1,0), 0).x;
    float c01 = texelFetch(nSampler, p+ivec2(0,1), 0).x;
    float c11 = texelFetch(nSampler, p+ivec2(1,1), 0).x;
    float c = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
    return c;
}

void main() {
    vec2 uv = (gl_FragCoord.xy+0.5)/iResolution.xy;

    vec3 col = texture(iSampler, vec2(uv.x,1.0-uv.y)).xyz;

    float alpha = 0.0;
    vec2 sc = min(iResolution.x, iResolution.y) / iResolution.xy;
    vec2 s0 = 0.01*sc, s1 = 0.011*sc;
    const float n = 32.;
    for (float i = 0.; i < n; i++) {
        float t = 2.0*3.14159*i/n;
        vec2 r = vec2(cos(t),sin(t));
        float a1 = getN(uv+r*s1);
        float a2 = getN(uv+r*s0);
        float a = 5.0*(a1-a2);
        a = clamp(a, 0.0, 1.0);
        // a = pow(a, 2.0) / (pow(1.0-a,2.0)+pow(a,2.0));
        a = 0.5*(a+a*a);
        alpha = max(alpha, a);
    }

    vec2 border = min(uv, 1.0-uv) - s1;
    if (min(border.x, border.y) > 1.0/min(iResolution.x,iResolution.y) || false) {
        col = mix(vec3(0.6,0.55,0.6)*col, vec3(0.5,1,0), alpha);
    }

    fragColor = vec4(col, 1);
}

