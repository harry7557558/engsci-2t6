precision highp float;

attribute vec4 aVertexPosition;

uniform float iDist;
uniform vec2 iResolution;
uniform mat4 iTransform;

varying vec3 vPos;

void main(void) {

    vec2 xy = aVertexPosition.xy;
    xy *= cos(0.5*asin(sin(2.0*atan(xy.y,xy.x))));
    xy = 2.0 * iDist * xy / (1.25-length(xy));
    float z = beach(xy).w;
    vPos = vec3(xy, z);
    vec4 pos = vec4(vPos, 1.0);

    gl_Position = iTransform * pos;
}
