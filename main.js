const btn = document.getElementsByClassName("button")
const canvasElement = document.getElementById('mediapipe_canvas');
const canvasCtx = canvasElement.getContext('2d');
const homeURL = window.location.origin
const simulation = document.querySelector(".simulation_header")

let landmarks_var = ""

const delay = ms => new Promise(res => setTimeout(res, ms));

async function load_model() {
    const model = await tf.loadLayersModel(`${homeURL}/model_tf/model.json`)
    return model
}

const model = load_model()


function onResults(results) {
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    if (results.multiFaceLandmarks) {
        for (const landmarks of results.multiFaceLandmarks) {
            landmarks_var = landmarks
            drawConnectors(canvasCtx, landmarks, FACEMESH_TESSELATION, {color: '#C0C0C070', lineWidth: 1});
            drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYE, {color: '#FF3030'});
            drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYEBROW, {color: '#FF3030'});
            drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_IRIS, {color: '#FF3030'});
            drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYE, {color: '#30FF30'});
            drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYEBROW, {color: '#30FF30'});
            drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_IRIS, {color: '#30FF30'});
            drawConnectors(canvasCtx, landmarks, FACEMESH_FACE_OVAL, {color: '#E0E0E0'});
            drawConnectors(canvasCtx, landmarks, FACEMESH_LIPS, {color: '#E0E0E0'});
        }
    }
    canvasCtx.restore();
}

const faceMesh = new FaceMesh({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
}});

faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

faceMesh.onResults(onResults);

let intervalID_global = ""
let count_interval = 0

async function interval(id) {
    var canvas = document.getElementById("frame_canvas");
    var canvas_test = document.getElementById("frame_canvas_hidden");
    let vid_test = document.getElementById("video_sel_hidden")

    if (count_interval > vid_test.duration) {
        clearInterval(id)
    }

    canvas_test.getContext("2d").drawImage(vid_test, 0, 0, 1080, 1920);
    canvas.getContext("2d").drawImage(canvas_test, 0, 0, 250, 470);
    await faceMesh.send({image: canvas_test})
    if (landmarks_var !== "") {
        const get_image_pyscript = pyscript.runtime.globals.get('get_image')
        get_image_pyscript(canvas_test.toDataURL(), landmarks_var)
    }
    count_interval += 1
}

async function handle_click(e) {
    landmarks_var = ""
    const id = e.target.id
    let vid = document.getElementById("video_sel")
    let vid_test = document.getElementById("video_sel_hidden")

    vid.src= `${homeURL}/Media/${id}.mp4`
    vid_test.src= `${homeURL}/Media/${id}.mp4`
    vid_test.play();

    const intervalID = setInterval(() => {
        intervalID_global = intervalID
        interval(intervalID)
    }, 1000)
}

function update_canvas(srcL, srcR) {
    const leftImg = document.getElementById("leftImg")
    const rightImg = document.getElementById("rightImg")
    

    leftImg.src=srcL
    rightImg.src=srcR

    predict()
}

function eval_pred(predL, predR) {

    const pL = document.getElementById("leftpred")
    const pR = document.getElementById("rightpred")
    const pO = document.getElementById("overallpred")

    const alertL = predL[0]
    const drowsyL = predL[1]

    const alertR = predR[0]
    const drowsyR = predR[1]

    if ((drowsyL > 0.75) && (drowsyR > 0.75)) {
        pO.innerHTML = "Drowsy"
    } else {
        pO.innerHTML = "Alert"
    }

    if (drowsyL > 0.75) {
        pL.innerHTML = `Drowsy`
    } else {
        pL.innerHTML = `Alert`
    }

    if (drowsyR > 0.75) {
        pR.innerHTML = `Drowsy`
    } else {
        pR.innerHTML = `Alert`
    }

    
}

function predict() {
    const leftImg = document.getElementById("leftImg")
    const rightImg = document.getElementById("rightImg")

    model.then(async function (res) {
        const imgLPP = preprocess(leftImg);
        const imgRPP = preprocess(rightImg)
        const predL = res.predict(imgLPP[0]).dataSync();
        const predR = res.predict(imgRPP[0]).dataSync();
        eval_pred(predL, predR)
    }, function (err) {
        console.log(err);
    });
}


function preprocess(img) {
    let tensor = tf.browser.fromPixels(img)
    let resized = tf.image.resizeBilinear(tensor, [40, 85]).toFloat()
    resized = resized.div(255.0)
    const reshape = resized.reshape([1, 40, 85, 3])


    return [reshape, resized]
}


btn.forEach(element => {
    element.addEventListener("click", (e) => {
        count_interval = 0
        simulation.style.display="block"
        if (intervalID_global === "") {
            handle_click(e)
        } else {
            clearInterval(intervalID_global)
            handle_click(e)
        }
    })
});

