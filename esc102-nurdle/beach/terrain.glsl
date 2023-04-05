// #include <noise.glsl>

#define noise  SimplexNoise2D

float smin_h;
vec4 smin(vec4 a, vec4 b, float k) {
    smin_h = clamp(0.5 + 0.5 * (b.w - a.w) / k, 0., 1.);
    float d = mix(b.w, a.w, smin_h) - k * smin_h * (1.0 - smin_h);
    return vec4(mix(b.xyz, a.xyz, smin_h), d);
}
vec4 smax(vec4 a, vec4 b, float k) {
    return smin(a*vec4(1,1,1,-1), b*vec4(1,1,1,-1), k) * vec4(1,1,1,-1);
}

float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p-a, ba = b-a;
    float h = clamp(dot(pa,ba)/dot(ba,ba), 0.0, 1.0);
    return length(pa - ba*h);
}



float softSaturate(float x, float k) {
	return x / length(vec2(x/k,1));
}


float nurdle_k = 0.0;

vec4 beach(vec2 xy0) {
	// base
	vec2 xy = xy0 + 0.5;
	float n = 0.7*noise(30.0*xy) + 0.3*noise(60.0*xy) + 0.2*noise(100.0*xy);
	vec4 base = vec4(
		mix(vec3(0.6,0.6,0.5), vec3(0.8,0.75,0.7), smoothstep(-0.1,0.1,n)),
		0.02*n);
	base.xyz *= 1.0+vec3(0.15,0,-0.15)*noise(vec2(0.8,1.2)*xy)+vec3(0.2,0,-0.2)*noise(vec2(0.6,0.3)*xy+5.);
	base.xyz = pow(base.xyz, vec3(1.5));

	// different sands
	xy += 0.5;
	n = noise(40.0*xy);
	vec4 sand = vec4(0.1,0.1,0.1, 0.5*(softSaturate(exp(20.0*(n-0.5)),0.02)-0.01));
	base = smax(base, sand, 0.005);
	xy += 0.5;
	n = noise(20.0*xy);
	sand = vec4(mix(vec3(0.7,0.3,0.2),vec3(0.7,0.6,0.4),smoothstep(-0.2,0.2,noise(40.*xy+1.))),
		0.5*(softSaturate(exp(20.0*(n-0.5)),0.02)-0.005));
	base = smax(base, sand, 0.005);

	// rocks
	xy += 0.5;
	n = min(noise(1.5*xy) + 0.2*noise(5.0*xy) + 0.05*noise(30.0*xy), 4.0*noise(0.25*xy));
	vec4 rock = vec4(
		pow(vec3(0.2,0.3,0.4),vec3(1.0+softSaturate(4.0*noise(1.0*xy+1.),1.))),
		softSaturate(exp(10.0*(n-0.5))-0.05,0.1));
	if (rock.w > 0.) rock.w = 0.25*sqrt(rock.w);
	base = smax(base, rock, 0.005);

	// wood
	xy += 1.2;
	vec2 xyg = 0.2*xy;
	for (int i = 0; i < 3; i++) {
	xyg = 1.5*mat2(0.6,-0.8,0.8,0.6)*xyg + 2.5;
	vec2 grid = floor(xyg+0.5);
	if (hash12(grid) < 0.2) {
		float a = 2.0*PI*hash12(grid+0.1);
		float l = 0.05+0.35*hash12(grid+0.2);
		float w = 0.005+0.02*hash12(grid+0.3);
		vec2 f = xyg-grid + 0.2*vec2(noise(xyg+0.5),noise(-xyg+1.5));
		float d = sdSegment(f, l*vec2(cos(a),sin(a)), -l*vec2(cos(a),sin(a)));
		d = w*w-d*d;
		if (d>0.0) d = 2.0*sqrt(d)+w;
		else d -= w;
		vec4 branch = vec4(0.4,0.3,0.2,d);
		branch.xyz *= mix(vec3(1,0.4,0.2),vec3(0.9,0.9,0.8),hash12(grid-0.1));
		branch.xyz = pow(branch.xyz, vec3(0.2+1.5*smoothstep(-0.5,0.5,noise(0.4*xyg))));
		base = smax(base, branch, 0.01);
	}
	}

	// nurdles
	xy = xy0;
	xyg = 2.0*xy;
	vec2 grid = floor(xyg+0.5);
	if (hash12(grid) < 0.1+0.1*sin(grid.x)*cos(grid.y)) {
		float x = -0.4+0.4*hash12(grid+0.1);
		float y = -0.4+0.4*hash12(grid+0.2);
		float ani = 0.1*pow(hash12(grid+0.3),2.);
		float a = 2.0*PI*hash12(grid+0.4);
		float s = 0.1+0.02*hash12(grid+0.5);
		float h = 0.2*s * hash12(grid+0.6);
		vec2 f = xyg-grid - vec2(x,y);
		float d = s*s - dot(f,f) / pow(1.0+ani*sin(atan(f.y,f.x)-a),2.);
		if (d > 0.0) d = 0.5*sqrt(d)-h;
		else d -= s;
		vec4 nurdle = vec4(0.7,0.7,0.7,d);
		// nurdle.xyz = vec3(1,0,1);
		base = smax(base, nurdle, 0.01);
		nurdle_k = max(nurdle_k, 1.0-smin_h);
	}

	return base;
}