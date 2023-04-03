precision highp float;

uniform float iRx, iRz, iRy;
uniform float iDist;
uniform vec2 iResolution;

varying vec3 vPos;

vec3 sundir = normalize(vec3(1,0.5,0.5));
vec3 suncol = 0.5*vec3(1,0.9,0.6);
vec3 skycol = 0.7*vec3(0.5,0.7,0.9);


void main() {
    vec3 col = vec3(1.0);

    vec2 xy = vPos.xy;
    vec2 grad = vec2(
        beach(xy+vec2(0.01,0)).w-beach(xy-vec2(0.01,0)).w,
        beach(xy+vec2(0,0.01)).w-beach(xy-vec2(0,0.01)).w)/0.02;
    vec3 n = normalize(vec3(grad, 1));

    nurdle_k = 0.0;
    vec3 albedo = beach(xy).xyz;
    col = 0.1 * albedo;
  
    col += suncol * max(0.2+0.8*dot(n,sundir),0.0) * albedo;
    col += skycol * max(n.z,0.0) * albedo;
    col += 0.1*max(-dot(n,sundir),0.0) * albedo;

    col = mix(col, vec3(1), smoothstep(3.0*iDist, 8.0*iDist, length(xy)));

    col = pow(col, vec3(0.4545));

    gl_FragColor = vec4(col, 0.5+0.5*nurdle_k);
    // gl_FragColor = vec4(vec3(nurdle_k), 1);
}
