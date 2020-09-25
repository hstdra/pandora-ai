const video = document.getElementById("video");
const textData = [];
var model;

document.getElementById("start").onclick = function (event) {
  Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
    faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
    faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
    // faceapi.nets.faceExpressionNet.loadFromUri("/models"),
  ]).then(startVideo);
};

document.getElementById("btn-save").onclick = function (event) {
  saveData(JSON.stringify(textData));
};

function startVideo() {
  navigator.getUserMedia(
    {
      video: {
        width: 1280,
        height: 720,
        frameRate: 20,
      },
    },
    (stream) => (video.srcObject = stream),
    (err) => console.error(err)
  );
}

video.addEventListener("play", async () => {
  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);

  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  setInterval(async () => {
    const detections = await faceapi
      .detectSingleFace(
        video,
        new faceapi.TinyFaceDetectorOptions({
          inputSize: 320,
          scoreThreshold: 0.6,
        })
      )
      .withFaceLandmarks();

    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    const mouth = resizedDetections.landmarks.getMouth();

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // ctx.fillRect(mouth[0].x, mouth[0].y, 3, 3);
    // ctx.fillRect(mouth[1].x, mouth[1].y, 3, 3);
    // ctx.fillRect(mouth[2].x, mouth[2].y, 3, 3);
    // ctx.fillRect(mouth[3].x, mouth[3].y, 3, 3);
    // ctx.fillRect(mouth[4].x, mouth[4].y, 3, 3);
    // ctx.fillRect(mouth[5].x, mouth[5].y, 3, 3);
    // ctx.fillRect(mouth[6].x, mouth[6].y, 3, 3);
    // ctx.fillRect(mouth[7].x, mouth[7].y, 3, 3);
    // ctx.fillRect(mouth[8].x, mouth[8].y, 3, 3);
    // ctx.fillRect(mouth[9].x, mouth[9].y, 3, 3);

    // ctx.fillRect(mouth[10].x, mouth[10].y, 3, 3);
    // ctx.fillRect(mouth[11].x, mouth[11].y, 3, 3);
    // ctx.fillRect(mouth[12].x, mouth[12].y, 3, 3);

    ctx.fillRect(mouth[13].x, mouth[13].y, 3, 3);
    ctx.fillRect(mouth[14].x, mouth[14].y, 3, 3);
    ctx.fillRect(mouth[15].x, mouth[15].y, 3, 3);

    // ctx.fillRect(mouth[16].x, mouth[16].y, 3, 3);

    ctx.fillRect(mouth[17].x, mouth[17].y, 3, 3);
    ctx.fillRect(mouth[18].x, mouth[18].y, 3, 3);
    ctx.fillRect(mouth[19].x, mouth[19].y, 3, 3);

    // faceapi.draw.drawDetections(canvas, resizedDetections);
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
  }, 10000000);
  await loadModel();
  console.log("START...");
  await isTalking();

  // setInterval(async () => {
  //   isTalking();
  // }, 0);
  while (true) {
    await isTalking();

    await new Promise((resolve) => {
      setTimeout(resolve, 100);
    });
  }
});

async function loadModel() {
  try {
    model = await tf.loadLayersModel(
      "http://localhost:3000/mouth-model/model.json"
    );
    console.log("Model loaded!");
  } catch (error) {
    console.log("loadModel -> error", error);
  }
}

function predict(openScores) {
  if (openScores.length < 20) return 0;

  const [acc] = model.predict(tf.tensor2d([openScores])).arraySync();
  console.log("predict -> acc", acc[0]);
}

function generateTrainData(openScores, isTalking) {
  if (openScores.length < 20) return;

  const data = {
    openScores,
    isTalking,
    std: math.std(openScores),
    mean: math.mean(openScores),
  };

  textData.push(data);
  // console.log("generateTrainData -> textData", textData);
}

function calculateTalking(openScores) {
  const std = math.std(openScores);
  const mean = math.mean(openScores);
  const isTalking = std > 0.1 && mean > 0.22 ? 1 : 0;

  return isTalking;
}

async function isTalking() {
  const openScores = [];
  // const mouthPoints = [];

  let promise = Promise.resolve();

  for (let index = 0; index < 20; index++) {
    try {
      const mouth = await getMouth();
      const score = getMouthOpenScore(mouth);

      openScores.push(score);
      // addMouthPoints(mouthPoints, mouth);
    } catch (error) {}

    promise = promise.then(function () {
      return new Promise(function (resolve) {
        setTimeout(resolve, 50);
      });
    });
  }

  return promise.then(function () {
    try {
      // const isTalking = calculateTalking(openScores);
      // console.log(isTalking ? "TALKING" : "NONE");

      predict(openScores);
      generateTrainData(openScores, isTalking);
    } catch (error) {}
  });
}

async function getMouth() {
  const detections = await faceapi
    .detectSingleFace(
      video,
      new faceapi.TinyFaceDetectorOptions({
        inputSize: 320,
        scoreThreshold: 0.6,
      })
    )
    .withFaceLandmarks();
  const displaySize = { width: video.width, height: video.height };

  const resizedDetections = faceapi.resizeResults(detections, displaySize);
  return resizedDetections.landmarks.getMouth();
}

function getMouthOpenScore(mouth) {
  const topDistance = math.distance([
    getCordinate(mouth[13]),
    getCordinate(mouth[15]),
  ]);
  const bottomDistance = math.distance([
    getCordinate(mouth[17]),
    getCordinate(mouth[19]),
  ]);
  const topDownDistance = math.distance([
    getCordinate(mouth[14]),
    getCordinate(mouth[18]),
  ]);

  return (topDownDistance * 2) / (+topDistance + +bottomDistance);
}

function getCordinate(point) {
  return [point.x, point.y];
}

function saveData(text) {
  const filename = "file.json";
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  saveAs(blob, filename);
}
