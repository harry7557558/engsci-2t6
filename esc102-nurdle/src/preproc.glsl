#version 300 es
precision highp float;

out vec4 fragColor;

uniform vec2 iResolution;

uniform sampler2D iSampler;
uniform vec3 bnMu, bnVar, bnA, bnB;  // batch norm, fixed

void main() {
    vec2 uv = (gl_FragCoord.xy+0.5)/iResolution.xy;
    uv.y = 1.0-uv.y;
    vec3 s = texture(iSampler, uv).xyz;
    s = (s-bnMu)/sqrt(bnVar+1e-5) * bnA + bnB;
    s = max(s, 0.2*s);
    fragColor = vec4(s, 1);
}
