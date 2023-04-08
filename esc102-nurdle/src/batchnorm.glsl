#version 300 es
precision highp float;

out vec4 fragColor;

uniform int ZERO;

uniform vec2 iResolution;
uniform sampler2D iSampler;

uniform vec4 bnMu, bnVar, bnA, bnB;


void meanVar(vec2 uv, out vec4 mean, out vec4 var) {
    vec2 sc = min(iResolution.x, iResolution.y) / iResolution.xy;
    const float n = 128.0;
    vec4 s1 = vec4(0), s2 = vec4(0);
    for (float i = 0.5; i < n; i++) {
        const float phi = 0.5*(1.+sqrt(5.));
        float u1 = mod(i/phi, 1.0), u2 = i/n;
        float a = 2.0*3.1415923*u2;
        float r = 0.2*sqrt(-2.0*log(1.0-u1));
        vec4 s = texture(iSampler, uv + r*sc * vec2(cos(a),sin(a)));
        s1 += s, s2 += s*s;
    }
    mean = s1/n;
    var = (s2-s1*s1/n)/(n-1.);
}

void main() {
    vec2 uv = gl_FragCoord.xy / iResolution.xy;
    vec4 s = texture(iSampler, uv);

    vec4 mean = bnMu, var = bnVar;
    // meanVar(uv, mean, var);
    s = (s-mean)/sqrt(var+1e-5) * bnA + bnB;
    s = max(s, 0.2*s);
    fragColor = s;
}
