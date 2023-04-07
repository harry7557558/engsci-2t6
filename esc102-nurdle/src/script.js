"use strict";


// rendering related
var renderer = {
    canvas: null,
    gl: null,
    nLayers: 12,
    width: -1,
    height: -1,
    iFrame: 0,
    renderNeeded: true
};

function i2id(i, d = 2) {
    i = i.toString();
    while (i.length < d) i = "0" + i;
    return i;
}

// hard-coded batch norm weights
const weights = {
    bn000: [1.0875437, 0.5369309, 1.473317],
    bn001: [0.2487396, -0.9525007, -0.19628945],
    bn002: [0.5844249, 0.5447748, 0.49758208],
    bn003: [0.030239817, 0.028824896, 0.030168401],
    bn010: [1.2324575, 0.7495642, 1.381774, 1.3054899],
    bn011: [0.19467758, 0.19788893, 0.17214468, 0.22925404],
    bn012: [-0.8799407, -2.5948296, -0.6307882, 0.13043472],
    bn013: [1.4815185, 4.076296, 2.7199693, 5.0037313],
    bn020: [0.6682743, 1.5771377, 1.0267625, 1.2508844],
    bn021: [-0.43936253, 0.42591658, -0.6834548, 0.16599116],
    bn022: [2.5228357, 0.7737249, -0.43457204, 0.027493501],
    bn023: [5.16551, 8.782018, 2.882289, 7.2866917],
    bn030: [1.6136072, 1.1954765, 1.0695794, 0.9225741],
    bn031: [0.14704981, -0.42891672, -0.48235574, -0.19320083],
    bn032: [-2.8392692, -2.6562238, -1.1566317, 3.1849725],
    bn033: [24.423037, 25.105013, 8.510271, 28.97101],
    bn040: [1.3041219, 0.69821906, 0.66234356, 1.1005255],
    bn041: [0.751065, -0.005547544, -0.6242232, 1.0371691],
    bn042: [-0.89758086, -0.32209155, -0.9474234, -0.6896971],
    bn043: [5.2010565, 16.417025, 8.812019, 4.848618],
    bn050: [0.8799694, 1.0538652, 1.1641568, 1.200811],
    bn051: [-0.27644825, -0.3896804, 0.59128076, 0.30521405],
    bn052: [-4.670373, -0.85437447, -2.345801, -1.4016482],
    bn053: [18.855791, 11.491445, 19.035099, 6.4028406],
    bn060: [0.8946732, 0.9071275, 1.007496, 1.199179],
    bn061: [0.5000499, -0.47228158, 0.28620112, -0.7111842],
    bn062: [1.9119624, 2.6707883, -4.6874204, 0.1788203],
    bn063: [4.190943, 9.364522, 21.634615, 8.907826],
    bn070: [1.1327761, 1.2169952, 1.0438268, 1.2242298],
    bn071: [-0.29658198, -0.4025115, -0.16613413, -0.021814192],
    bn072: [0.25966427, 2.2204158, -0.49577796, -0.72139126],
    bn073: [11.897299, 11.092597, 5.0902762, 17.471506],
    bn080: [1.6280907, 0.976057, 1.3802146, 0.7956999],
    bn081: [-0.0999923, 0.45473814, -0.19672333, -0.009975298],
    bn082: [-0.25645903, -1.7288702, 0.26522893, 1.5673889],
    bn083: [42.53321, 4.855316, 30.56159, 10.939028],
    bn090: [0.9479627, 0.76004285, 0.9669737, 1.297626],
    bn091: [-0.3172532, 0.14124908, 0.5320815, 0.8525788],
    bn092: [-0.16991317, -2.0539656, 0.14125739, -2.3391674],
    bn093: [3.6389396, 13.038954, 33.76478, 19.781153],
    bn100: [1.3477169, 1.1263705, 0.62507266, 0.5132608],
    bn101: [0.94958377, -0.34217754, -0.18692152, -0.5332354],
    bn102: [3.7211475, 3.7108977, -4.0354342, 5.73437],
    bn103: [24.345665, 30.162807, 13.131075, 31.46819],
    bn110: [0.8656952, 0.9007957, 0.5685502, 0.31277353],
    bn111: [0.49168396, 0.52483743, 0.75863594, 0.25391576],
    bn112: [-5.156895, 5.2432404, 3.7451725, -6.6363025],
    bn113: [20.904793, 33.036736, 31.335789, 52.296574],
    
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
    image.onload = function () {
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
            srcFormat, srcType, image);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        callback(image.width, image.height);
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
    setBN(renderer.preprocProgram, "bnMu", weights.bn002);
    setBN(renderer.preprocProgram, "bnVar", weights.bn003);
    setBN(renderer.preprocProgram, "bnA", weights.bn000);
    setBN(renderer.preprocProgram, "bnB", weights.bn001);
    setPositionBuffer(renderer.preprocProgram);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // convolution
    gl.useProgram(renderer.convProgram);
    for (var i = 0; i < renderer.nLayers; i++) {
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
    gl.bindTexture(gl.TEXTURE_2D, renderer.image);
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

    // position buffer
    renderer.positionBuffer = renderer.gl.createBuffer();
    renderer.gl.bindBuffer(renderer.gl.ARRAY_BUFFER, renderer.positionBuffer);
    var positions = [-1, 1, 1, 1, -1, -1, 1, -1];
    renderer.gl.bufferData(renderer.gl.ARRAY_BUFFER, new Float32Array(positions), renderer.gl.STATIC_DRAW);

    // framebuffers
    renderer.preprocTarget = createRenderTarget(renderer.width, renderer.height);
    renderer.convTargets = [];
    for (var i = 0; i < renderer.nLayers; i++)
        renderer.convTargets.push(createRenderTarget(renderer.width, renderer.height));

    function updateRendererSize(w, h) {
        renderer.width = canvas.width = w;
        renderer.height = canvas.height = h;
        destroyRenderTarget(renderer.preprocTarget);
        renderer.preprocTarget = createRenderTarget(w, h);
        for (var i = 0; i < renderer.nLayers; i++) {
            destroyRenderTarget(renderer.convTargets[i]);
            renderer.convTargets[i] = createRenderTarget(w, h);
        }
    }

    // image - do this earlier
    var imgid = 0;
    var imgw = 600;
    var loadTextureCallback = function(w, h) {
        var sc = imgw / Math.max(w, h);
        updateRendererSize(sc*w, sc*h);
        renderer.renderNeeded = true;
    };
    renderer.image = loadTexture(
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
        renderer.image = loadTexture(
            "train/images/train/" + i2id(imgid) + ".jpg",
            loadTextureCallback
        );
    });

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

    // rendering
    function render() {
        drawScene();
        renderer.iFrame += 1;
        setTimeout(function () { requestAnimationFrame(render); }, 100);
    }
    requestAnimationFrame(render);

    // interactions
    window.addEventListener("resize", function (event) {
    });

}