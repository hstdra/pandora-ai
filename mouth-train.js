const threshold = 30;
const textData = [];

const video = document.getElementById("video");
const startButton = document.getElementById("start");
const saveButton = document.getElementById("btn-save");
const landmarksButton = document.getElementById("landmarks");

let canvas,
  displaySize,
  resizedDetections,
  isShowLandmarks = true;

startButton.onclick = async function () {
  startButton.style.display = "none";

  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri("./models"),
    faceapi.nets.faceLandmark68Net.loadFromUri("./models"),
    faceapi.nets.faceRecognitionNet.loadFromUri("./models"),
    // faceapi.nets.faceExpressionNet.loadFromUri("/models"),
  ]);
  startVideo();

  saveButton.style.display = "unset";
  landmarksButton.style.display = "unset";
};

saveButton.onclick = function () {
  saveData(JSON.stringify(textData));
};

landmarksButton.onclick = function () {
  isShowLandmarks = !isShowLandmarks;
  canvas.style.display = isShowLandmarks ? "unset" : "none";
};

function startVideo() {
  navigator.getUserMedia(
    {
      video: {
        width: 640,
        height: 360,
        frameRate: 30,
      },
    },
    (stream) => (video.srcObject = stream),
    (err) => console.error(err)
  );
}

video.addEventListener("play", async () => {
  canvas = faceapi.createCanvasFromMedia(video);
  document.getElementById("content").append(canvas);

  displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  // setInterval(async () => {
  //   try {
  //     const detections = await faceapi
  //       .detectSingleFace(
  //         video,
  //         new faceapi.TinyFaceDetectorOptions({
  //           inputSize: 320,
  //           scoreThreshold: 0.6,
  //         })
  //       )
  //       .withFaceLandmarks();

  //     const resizedDetections = faceapi.resizeResults(detections, displaySize);
  //     const mouth = resizedDetections.landmarks.getMouth();
  //   } catch (error) {}
  // }, 50);

  await isTalking();

  while (true) {
    await isTalking();

    await new Promise((resolve) => {
      setTimeout(resolve, 60);
    });
  }
});

function drawFaceToCanvas(mouth) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillRect(mouth[13].x, mouth[13].y, 3, 3);
  ctx.fillRect(mouth[14].x, mouth[14].y, 3, 3);
  ctx.fillRect(mouth[15].x, mouth[15].y, 3, 3);

  ctx.fillRect(mouth[17].x, mouth[17].y, 3, 3);
  ctx.fillRect(mouth[18].x, mouth[18].y, 3, 3);
  ctx.fillRect(mouth[19].x, mouth[19].y, 3, 3);

  faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
}

function generateTrainData(openScores, isTalking) {
  if (openScores.length < threshold) return;

  const data = {
    openScores,
    isTalking,
    std: math.std(openScores),
    mean: math.mean(openScores),
  };

  textData.push(data);
  // console.log("generateTrainData -> textData", textData);

  document.getElementById("normal").innerText = `Total NORMAL: ${
    textData.filter((x) => x.isTalking === 0).length
  }`;
  document.getElementById("talking").innerText = `Total TALKING: ${
    textData.filter((x) => x.isTalking === 1).length
  }`;
  document.getElementById("status").innerText = `Last Status: ${
    isTalking === 1 ? "TALKING" : "NORMAL"
  } - ${new Date().toTimeString()}`;
}

function calculateTalking(openScores) {
  const std = math.std(openScores);
  const mean = math.mean(openScores);
  const isTalking = std > 0.1 && mean > 0.22 ? 1 : 0;

  return isTalking;
}

async function isTalking() {
  const openScores = [];

  let promise = Promise.resolve();

  for (let index = 0; index < threshold; index++) {
    try {
      const mouth = await getMouth();
      const score = getMouthOpenScore(mouth);

      openScores.push(score);
      drawFaceToCanvas(mouth);
    } catch (error) {}

    promise = promise.then(function () {
      return new Promise(function (resolve) {
        setTimeout(resolve, 48);
      });
    });
  }

  return promise.then(function () {
    try {
      const isTalking = calculateTalking(openScores);
      console.log(isTalking ? "TALKING" : "NORMAL");

      generateTrainData(openScores, isTalking);
    } catch (error) {}
  });
}

async function getMouth() {
  const detections = await faceapi
    .detectSingleFace(
      video,
      new faceapi.TinyFaceDetectorOptions({
        inputSize: 224,
        scoreThreshold: 0.5,
      })
    )
    .withFaceLandmarks();
  displaySize = { width: video.width, height: video.height };

  resizedDetections = faceapi.resizeResults(detections, displaySize);
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
