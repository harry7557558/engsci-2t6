"use strict";


// rendering related
var renderer = {
    canvas: null,
    gl: null,
    nLayers: 12,
    width: -1,
    height: -1,
    renderVideo: true,
    image: null,
    imageTexture: null,
    renderNeeded: true
};

function i2id(i, d = 2) {
    i = i.toString();
    while (i.length < d) i = "0" + i;
    return i;
}


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
function loadTexture(url, callback) {
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
    image.onload = function() {
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
            srcFormat, srcType, image);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        callback(image);
    };
    image.onerror = function() {
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
    req.onerror = function(e) {
        alert("Failed to load texture " + texture_name);
    };
    req.onload = function(e) {
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
    if (!renderer.renderNeeded)
        return;
    renderer.renderNeeded = false;

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
        bn = bn.slice();
        while (bn.length < 4) bn.push(0);
        gl.uniform4f(location, bn[0], bn[1], bn[2], bn[3]);
    }

    // update video
    gl.bindTexture(gl.TEXTURE_2D, renderer.imageTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, renderer.image);

    // preprocessing
    gl.useProgram(renderer.preprocProgram);
    gl.bindFramebuffer(gl.FRAMEBUFFER, renderer.preprocTarget.framebuffer);
    //gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, renderer.imageTexture);
    gl.uniform1i(gl.getUniformLocation(renderer.preprocProgram, "iSampler"), 0);
    gl.uniform2f(gl.getUniformLocation(renderer.preprocProgram, "iResolution"),
        renderer.width, renderer.height);
    setBN(renderer.preprocProgram, "bnMu", weights.bn002);
    setBN(renderer.preprocProgram, "bnVar", weights.bn003);
    setBN(renderer.preprocProgram, "bnA", weights.bn000);
    setBN(renderer.preprocProgram, "bnB", weights.bn001);
    setPositionBuffer(renderer.preprocProgram);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    //return;

    for (var i = 0; i < renderer.nLayers; i++) {
        // batch norm
        // gl.useProgram(renderer.bnProgram);
        // gl.bindFramebuffer(gl.FRAMEBUFFER,
        //     renderer.bnTargets[i].framebuffer);
        // gl.uniform1i(gl.getUniformLocation(renderer.bnProgram, "ZERO"), 0);
        // gl.uniform2f(gl.getUniformLocation(renderer.bnProgram, "iResolution"),
        //     renderer.width, renderer.height);

        // gl.activeTexture(gl.TEXTURE0);
        // gl.bindTexture(gl.TEXTURE_2D,
        //     i == 0 ? renderer.preprocTarget.texture : renderer.convTargets[i - 1].texture);
        // gl.uniform1i(gl.getUniformLocation(renderer.bnProgram, "iSampler"), 0);

        // var j = i2id(i);
        // setBN(renderer.bnProgram, "bnMu", weights['bn' + j + '2']);
        // setBN(renderer.bnProgram, "bnVar", weights['bn' + j + '3']);
        // setBN(renderer.bnProgram, "bnA", weights['bn' + j + '0']);
        // setBN(renderer.bnProgram, "bnB", weights['bn' + j + '1']);

        // setPositionBuffer(renderer.bnProgram);
        // gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        // convolution
        gl.useProgram(renderer.convProgram);
        gl.bindFramebuffer(gl.FRAMEBUFFER, renderer.convTargets[i].framebuffer);
        gl.uniform1i(gl.getUniformLocation(renderer.convProgram, "ZERO"), 0);
        gl.uniform2f(gl.getUniformLocation(renderer.convProgram, "iResolution"),
            renderer.width, renderer.height);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D,
            i == 0 ? renderer.preprocTarget.texture : renderer.convTargets[i - 1].texture);
        // gl.bindTexture(gl.TEXTURE_2D, renderer.bnTargets[i].texture);
        gl.uniform1i(gl.getUniformLocation(renderer.convProgram, "iSampler"), 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, renderer['w' + i2id(i)]);
        gl.uniform1i(gl.getUniformLocation(renderer.convProgram, "iW"), 1);

        gl.uniform1i(gl.getUniformLocation(renderer.convProgram, "nIn"),
            i == 0 ? 3 : 4);
        gl.uniform1i(gl.getUniformLocation(renderer.convProgram, "nOut"),
            i + 1 == renderer.nLayers ? 1 : 4);

        var j = i + 1;
        if (j < renderer.nLayers) {
            j = i2id(j);
            setBN(renderer.convProgram, "bnMu", weights['bn' + j + '2']);
            setBN(renderer.convProgram, "bnVar", weights['bn' + j + '3']);
            setBN(renderer.convProgram, "bnA", weights['bn' + j + '0']);
            setBN(renderer.convProgram, "bnB", weights['bn' + j + '1']);
        }

        setPositionBuffer(renderer.convProgram);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    // highlighting
    gl.useProgram(renderer.highlightProgram);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.uniform2f(gl.getUniformLocation(renderer.highlightProgram, "iResolution"),
        renderer.width, renderer.height);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, renderer.imageTexture);
    gl.uniform1i(gl.getUniformLocation(renderer.highlightProgram, "iSampler"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, renderer.convTargets[renderer.nLayers - 1].texture);
    gl.uniform1i(gl.getUniformLocation(renderer.highlightProgram, "nSampler"), 1);
    setPositionBuffer(renderer.highlightProgram);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

}


// load model
function loadModel() {
    let path = "train/weights";
    loadWeightTexture(path + "/w00_4_3_3_3.bin", "w00");
    loadWeightTexture(path + "/w01_4_4_3_3.bin", "w01");
    loadWeightTexture(path + "/w02_4_4_3_3.bin", "w02");
    loadWeightTexture(path + "/w03_4_4_3_3.bin", "w03");
    loadWeightTexture(path + "/w04_4_4_3_3.bin", "w04");
    loadWeightTexture(path + "/w05_4_4_3_3.bin", "w05");
    loadWeightTexture(path + "/w06_4_4_3_3.bin", "w06");
    loadWeightTexture(path + "/w07_4_4_3_3.bin", "w07");
    loadWeightTexture(path + "/w08_4_4_3_3.bin", "w08");
    loadWeightTexture(path + "/w09_4_4_3_3.bin", "w09");
    loadWeightTexture(path + "/w10_4_4_3_3.bin", "w10");
    loadWeightTexture(path + "/w11_1_4_3_3.bin", "w11");
}

// load renderer/interaction
window.onload = function() {
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
    canvas.addEventListener("webglcontextlost", function(event) {
        event.preventDefault();
        onError("Error: WebGL context lost.");
    }, false);
    renderer.width = canvas.width;
    renderer.height = canvas.height;
    let gl = renderer.gl;

    // position buffer
    renderer.positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, renderer.positionBuffer);
    var positions = [-1, 1, 1, 1, -1, -1, 1, -1];
    gl.bufferData(renderer.gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    // framebuffers
    renderer.preprocTarget = createRenderTarget(renderer.width, renderer.height);
    renderer.convTargets = [];
    // renderer.bnTargets = [];
    for (var i = 0; i < renderer.nLayers; i++) {
        renderer.convTargets.push(createRenderTarget(renderer.width, renderer.height));
        // renderer.bnTargets.push(createRenderTarget(renderer.width, renderer.height));
    }

    function updateRendererSize(w, h) {
        canvas.style.width = w + "px";
        canvas.style.height = h + "px";
        var sc1 = Math.min(1920.0 / Math.max(w, h), 1.0);
        var sc2 = Math.max(480.0 / Math.min(w, h), 1.0);
        var sc = Math.sqrt(sc1*sc2);
        w = Math.round(w * sc);
        h = Math.round(h * sc);
        renderer.width = canvas.width = w;
        renderer.height = canvas.height = h;
        destroyRenderTarget(renderer.preprocTarget);
        renderer.preprocTarget = createRenderTarget(w, h);
        for (var i = 0; i < renderer.nLayers; i++) {
            destroyRenderTarget(renderer.convTargets[i]);
            renderer.convTargets[i] = createRenderTarget(w, h);
            // destroyRenderTarget(renderer.bnTargets[i]);
            // renderer.bnTargets[i] = createRenderTarget(w, h);
        }
        renderer.renderNeeded = true;
    }
    if (renderer.renderVideo)
        updateRendererSize(window.innerWidth, window.innerHeight);
    else
        updateRendererSize(400, 900);

    // image testing - do this earlier
    if (!renderer.renderVideo) {
        var imgid = 0;
        var imgw = 600;
        var loadTextureCallback = function(image) {
            renderer.image = image;
            let w = image.width, h = image.height;
            let sc = imgw / Math.max(w, h);
            updateRendererSize(sc*w, sc*h);
            renderer.renderNeeded = true;
        };
        renderer.imageTexture = loadTexture(
            "train/images/train/00.jpg",
            loadTextureCallback
        );
        window.addEventListener("wheel", function(event) {
            if (event.shiftKey)
                return;
            event.preventDefault();
            if (event.ctrlKey)
                imgw *= Math.exp(event.deltaY > 0 ? -0.1 : 0.1)
            else
                imgid = (imgid + (event.deltaY > 0 ? 1 : 20)) % 21;
            renderer.imageTexture = loadTexture(
                "train/images/train/" + i2id(imgid) + ".jpg",
                loadTextureCallback
            );
        }, { passive: false });
    }

    // weights/textures
    loadModel();

    // GLSL source
    console.time("load glsl code");
    let vsSource = "#version 300 es\nin vec4 vertexPosition;" +
        "void main(){gl_Position=vertexPosition;}";
    let preprocSource = loadShaderSource("src/preproc.glsl");
    let convSource = loadShaderSource("src/conv.glsl");
    // let bnSource = loadShaderSource("src/batchnorm.glsl");
    let highlightSource = loadShaderSource("src/highlight.glsl");
    console.timeEnd("load glsl code");

    // shaders
    console.time("compile shader");
    try {
        renderer.preprocProgram = createShaderProgram(vsSource, preprocSource);
        renderer.convProgram = createShaderProgram(vsSource, convSource);
        // renderer.bnProgram = createShaderProgram(vsSource, bnSource);
        renderer.highlightProgram = createShaderProgram(vsSource, highlightSource);
    }
    catch (e) {
        return onError(e);
    }
    console.timeEnd("compile shader");

    // video
    if (renderer.renderVideo) {
        renderer.imageTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, renderer.imageTexture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        renderer.image = document.createElement('video');
        navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
                width: { min: 0, ideal: window.innerHeight, max: 1920 },
                height: { min: 0, ideal: window.innerWidth, max: 1920 },
                facingMode: "environment"
            },
        })
            .then(stream => {
                renderer.image.srcObject = stream;
                renderer.image.play();
                renderer.image.addEventListener('playing', () => {
                    renderer.renderNeeded = true;
                });

            })
            .catch(error => {
                // alert("Failed to access camera.");
                document.write('Error accessing camera: ', error);
                renderer.gl = null;
            });
    }

    // rendering
    function render() {
        if (renderer.gl == null)
            return;
        drawScene();
        if (renderer.renderVideo)
            renderer.renderNeeded = true;
        setTimeout(function() { requestAnimationFrame(render); }, 100);
    }
    requestAnimationFrame(render);

    // interactions
    window.addEventListener("resize", function(event) {
        if (renderer.renderVideo)
            updateRendererSize(window.innerWidth, window.innerHeight);
    });

}