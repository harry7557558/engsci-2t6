#version 300 es
precision highp float;

out vec4 fragColor;

uniform vec2 iResolution;

uniform sampler2D iSampler;
uniform vec4 iClip;  // xy bottom left uv, zw top right

uniform vec4 bnMu, bnVar, bnA, bnB;  // batch norm, fixed

vec3 getSampler(vec2 uv) {
    uv = mix(iClip.xy, iClip.zw, uv);
    uv.y = 1.0-uv.y;
    return texture(iSampler, uv).xyz;
}

void meanVar(vec2 uv, out vec3 mean, out vec3 var) {
    vec2 sc = min(iResolution.x, iResolution.y) / iResolution.xy;
    const float n = 24.0;
    vec3 s1 = vec3(0), s2 = vec3(0);
    for (float i = 0.5; i < n; i++) {
        const float phi = 0.5*(1.+sqrt(5.));
        float u1 = mod(i/phi, 1.0), u2 = i/n;
        float a = 2.0*3.1415923*u2;
        float r = 0.2*sqrt(-2.0*log(1.0-u1));
        vec3 s = getSampler(uv + r*sc * vec2(cos(a),sin(a)));
        s1 += s, s2 += s*s;
    }
    mean = s1/n;
    var = (s2-s1*s1/n)/(n-1.);
}

void main() {
    vec2 uv = (gl_FragCoord.xy+0.5)/iResolution.xy;
    vec3 s = getSampler(uv);
    vec3 mean, var;
    meanVar(uv, mean, var);
    s = (s-mean)/sqrt(var+1e-2);
    s = (s-bnMu.xyz)/sqrt(bnVar.xyz+1e-5);
    s = s * bnA.xyz + bnB.xyz;
    s = max(s, 0.2*s);
    fragColor = vec4(s, 1);
}
