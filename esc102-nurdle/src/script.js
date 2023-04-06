"use strict";


// rendering related
var renderer = {
    canvas: null,
    gl: null,
    width: -1,
    height: -1,
    iFrame: 0,
};

// hard-coded batch norm weights
const weights = {
    bn00: [1.5067475, 0.73462236, 0.9027463],
    bn01: [1.0160867, -0.5190285, 0.42915004],
    bn02: [0.6691655, 0.6620227, 0.6227424],
    bn03: [0.006436822, 0.0065827384, 0.007982741],
    bn10: [1.3520912, 0.8920276, 1.2923803, 1.0416291],
    bn11: [-0.37641674, -0.3698626, -0.06813239, -1.20616],
    bn12: [5.098454, -0.07922329, -1.5588161, 12.819611],
    bn13: [16.921711, 20.866167, 17.406403, 48.901127],
    bn20: [1.1458615, 0.9962743, 0.7491567, 1.1077212],
    bn21: [-1.4370297, -0.20699756, -1.573737, 0.25815344],
    bn22: [-2.7830803, 1.5557672, -4.217627, 1.8538486],
    bn23: [75.8165, 25.582893, 20.978085, 24.738379],
    bn30: [0.7774884, 1.4483851, 1.2628847, 0.7937931],
    bn31: [0.92373097, -0.033770848, 1.3037564, 0.2659414],
    bn32: [4.62958, 0.3990834, 5.1560297, -0.58347964],
    bn33: [22.434797, 27.502625, 30.352474, 19.078442],
    bn40: [0.95617354, 1.1717262, 1.0471597, 0.9096549],
    bn41: [1.3659971, 0.15213971, 0.7811826, 1.448209],
    bn42: [11.527573, -3.5671787, 6.5716677, 16.242535],
    bn43: [47.819103, 20.99566, 28.102276, 78.71919],
    bn50: [1],
    bn51: [0],
    bn52: [0],
    bn53: [1],
};

// request shader sources
function loadShaderSource(path) {
    var request = new XMLHttpRequest();
    request.open("GET", path, false);
    request.send(null);
    if (request.status != 200) return "";
    var source = request.responseText;
    return source;
}

// compile shaders and create a shader program
function createShaderProgram(vsSource, fsSource) {
    let gl = renderer.gl;
    function loadShader(gl, type, source) {
        var shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
            throw new Error("Shader compile error: " + gl.getShaderInfoLog(shader));
        return shader;
    }
    var vShader = loadShader(gl, gl.VERTEX_SHADER, vsSource);
    var fShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource);
    var shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vShader);
    gl.attachShader(shaderProgram, fShader);
    gl.linkProgram(shaderProgram);
    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS))
        throw new Error(gl.getProgramInfoLog(shaderProgram));
    return shaderProgram;
}

// image texture
function loadTexture(url) {
    let gl = renderer.gl;
    let texture = gl.createTexture();

    // temporary fill a white blank before the image finished download
    const level = 0;
    const internalFormat = gl.RGBA;
    const width = 1;
    const height = 1;
    const border = 0;
    const srcFormat = gl.RGBA;
    const srcType = gl.UNSIGNED_BYTE;
    const pixel = new Uint8Array([255, 255, 255, 255]);  // white
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
        width, height, border,
        srcFormat, srcType, pixel);

    const image = new Image();
    image.onload = function () {
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
            srcFormat, srcType, image);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    };
    image.onerror = function () {
        alert("Failed to load texture.");
    }
    // image.crossOrigin = "";
    image.src = url;
    return texture;
}


// create texture/framebuffer
function createSampleTexture(width, height) {
    let gl = renderer.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    const level = 0;
    const internalFormat = gl.RGBA32F;
    const border = 0;
    const format = gl.RGBA;
    const type = gl.FLOAT;
    const data = null;
    gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
        width, height, border,
        format, type, data);
    // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.MIRRORED_REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.MIRRORED_REPEAT);
    return tex;
}
function createRenderTarget(width, height) {
    let gl = renderer.gl;
    const tex = createSampleTexture(width, height);
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const sampler = createSampleTexture(gl, width, height);
    return {
        texture: tex,
        framebuffer: framebuffer,
        sampler: sampler
    };
}
function destroyRenderTarget(target) {
    let gl = renderer.gl;
    gl.deleteTexture(target.texture);
    gl.deleteFramebuffer(target.framebuffer);
}

function loadWeightTexture(url, texture_name) {
    const gl = renderer.gl;

    function onload(weights) {
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);

        var l = weights.length / 4, rl = Math.sqrt(l);
        var w = 1, h = l;
        for (var i = 1; i <= rl; i++) {
            var j = Math.round(l / i);
            if (i * j == l) w = i, h = j;
        }
        console.log(texture_name, w, h);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F,
            w, h, 0,
            gl.RGBA, gl.FLOAT,
            weights);

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_BASE_LEVEL, 0);
        // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, 0);

        renderer[texture_name] = tex;
    }

    var req = new XMLHttpRequest();
    req.open("GET", url, true);
    req.responseType = "arraybuffer";
    req.onerror = function (e) {
        alert("Failed to load texture " + texture_name);
    };
    req.onload = function (e) {
        if (req.status == 200) {
            var weights = new Float32Array(req.response);
            onload(weights);
        }
        else {
            req.onerror();
        }
    };
    req.send();
}


// call this function to re-render
async function drawScene() {
    let gl = renderer.gl;
    gl.viewport(0, 0, renderer.width, renderer.height);

    // set position buffer for vertex shader
    function setPositionBuffer(program) {
        var vpLocation = gl.getAttribLocation(program, "vertexPosition");
        const numComponents = 2;
        const type = gl.FLOAT;
        const normalize = false;
        const stride = 0, offset = 0;
        gl.bindBuffer(gl.ARRAY_BUFFER, renderer.positionBuffer);
        gl.vertexAttribPointer(
            vpLocation,
            numComponents, type, normalize, stride, offset);
        gl.enableVertexAttribArray(vpLocation);
    }

    // set batch norm uniforms
    function setBN(program, varname, bn) {
        let location = gl.getUniformLocation(program, varname);
        if (bn.length == 1)
            gl.uniform1f(location, bn[0]);
        if (bn.length == 2)
            gl.uniform2f(location, bn[0], bn[1]);
        if (bn.length == 3)
            gl.uniform3f(location, bn[0], bn[1], bn[2]);
        if (bn.length == 4)
            gl.uniform4f(location, bn[0], bn[1], bn[2], bn[3]);
    }

    // preprocessing
    gl.useProgram(renderer.preprocProgram);
    gl.bindFramebuffer(gl.FRAMEBUFFER, renderer.preprocTarget.framebuffer);
    // gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, renderer.image);
    gl.uniform1i(gl.getUniformLocation(renderer.preprocProgram, "iSampler"), 0);
    gl.uniform2f(gl.getUniformLocation(renderer.preprocProgram, "iResolution"),
        renderer.width, renderer.height);
    setBN(renderer.preprocProgram, "bnMu", weights.bn02);
    setBN(renderer.preprocProgram, "bnVar", weights.bn03);
    setBN(renderer.preprocProgram, "bnA", weights.bn00);
    setBN(renderer.preprocProgram, "bnB", weights.bn01);
    setPositionBuffer(renderer.preprocProgram);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // convolution
    gl.useProgram(renderer.convProgram);
    for (var i = 0; i < 5; i++) {
        gl.bindFramebuffer(gl.FRAMEBUFFER,
            renderer.convTargets[i].framebuffer);
        gl.uniform1i(gl.getUniformLocation(renderer.convProgram, "ZERO"), 0);
        gl.uniform2f(gl.getUniformLocation(renderer.convProgram, "iResolution"),
            renderer.width, renderer.height);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D,
            i == 0 ? renderer.preprocTarget.texture : renderer.convTargets[i-1].texture);
        gl.uniform1i(gl.getUniformLocation(renderer.convProgram, "iSampler"), 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, renderer['w0' + i]);
        gl.uniform1i(gl.getUniformLocation(renderer.convProgram, "iW"), 1);

        gl.uniform1i(gl.getUniformLocation(renderer.convProgram, "iKs"),
            [5, 5, 5, 3, 3][i]);
        gl.uniform1i(gl.getUniformLocation(renderer.convProgram, "nIn"),
            [3, 4, 4, 4, 4][i]);
        gl.uniform1i(gl.getUniformLocation(renderer.convProgram, "nOut"),
            [4, 4, 4, 4, 1][i]);

        var j = i + 1;
        setBN(renderer.convProgram, "bnMu", weights['bn' + j + '2']);
        setBN(renderer.convProgram, "bnVar", weights['bn' + j + '3']);
        setBN(renderer.convProgram, "bnA", weights['bn' + j + '0']);
        setBN(renderer.convProgram, "bnB", weights['bn' + j + '1']);

        setPositionBuffer(renderer.convProgram);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    // highlighting
    gl.useProgram(renderer.highlightProgram);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.uniform2f(gl.getUniformLocation(renderer.highlightProgram, "iResolution"),
        renderer.width, renderer.height);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, renderer.image);
    gl.uniform1i(gl.getUniformLocation(renderer.highlightProgram, "iSampler"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, renderer.convTargets[4].texture);
    gl.uniform1i(gl.getUniformLocation(renderer.highlightProgram, "nSampler"), 1);
    setPositionBuffer(renderer.highlightProgram);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

}


// load model
function loadModel() {
    let path = "train/weights";
    loadWeightTexture(path + "/w00_4_3_5_5.bin", "w00");
    loadWeightTexture(path + "/w01_4_4_5_5.bin", "w01");
    loadWeightTexture(path + "/w02_4_4_5_5.bin", "w02");
    loadWeightTexture(path + "/w03_4_4_3_3.bin", "w03");
    loadWeightTexture(path + "/w04_1_4_3_3.bin", "w04");
}

// load renderer/interaction
window.onload = function () {
    // get context
    function onError(error) {
        console.error(error);
        let errorMessageContainer = document.getElementById("error-message");
        errorMessageContainer.style.display = "block";
        errorMessageContainer.innerHTML = error;
    }
    let canvas = document.getElementById("canvas");
    renderer.canvas = canvas;
    renderer.gl = canvas.getContext("webgl2") || canvas.getContext("experimental-webgl2");
    if (renderer.gl == null)
        return onError("Error: Your browser may not support WebGL 2.0.");
    if (renderer.gl.getExtension("EXT_color_buffer_float") == null)
        return onError("Error: Your browser does not support WebGL float texture.");
    renderer.timerExt = renderer.gl.getExtension('EXT_disjoint_timer_query_webgl2');
    canvas.addEventListener("webglcontextlost", function (event) {
        event.preventDefault();
        onError("Error: WebGL context lost.");
    }, false);
    renderer.width = canvas.width;
    renderer.height = canvas.height;

    // image - do this first
    renderer.image = loadTexture("train/nurdles_train.jpg");

    // weights/textures
    loadModel();

    // GLSL source
    console.time("load glsl code");
    let vsSource = "#version 300 es\nin vec4 vertexPosition;" +
        "void main(){gl_Position=vertexPosition;}";
    let preprocSource = loadShaderSource("src/preproc.glsl");
    let convSource = loadShaderSource("src/conv.glsl");
    let highlightSource = loadShaderSource("src/highlight.glsl");
    console.timeEnd("load glsl code");

    // shaders
    console.time("compile shader");
    try {
        renderer.preprocProgram = createShaderProgram(vsSource, preprocSource);
        renderer.convProgram = createShaderProgram(vsSource, convSource);
        renderer.highlightProgram = createShaderProgram(vsSource, highlightSource);
    }
    catch (e) {
        return onError(e);
    }
    console.timeEnd("compile shader");

    // position buffer
    renderer.positionBuffer = renderer.gl.createBuffer();
    renderer.gl.bindBuffer(renderer.gl.ARRAY_BUFFER, renderer.positionBuffer);
    var positions = [-1, 1, 1, 1, -1, -1, 1, -1];
    renderer.gl.bufferData(renderer.gl.ARRAY_BUFFER, new Float32Array(positions), renderer.gl.STATIC_DRAW);

    // framebuffers
    renderer.preprocTarget = createRenderTarget(renderer.width, renderer.height);
    renderer.convTargets = [];
    for (var i = 0; i < 5; i++)
        renderer.convTargets.push(createRenderTarget(renderer.width, renderer.height));

    // rendering
    function render() {
        drawScene();
        renderer.iFrame += 1;
        // setTimeout(function () { requestAnimationFrame(render); }, 100);
    }
    requestAnimationFrame(render);

    // interactions
    window.addEventListener("resize", function (event) {
    });

}