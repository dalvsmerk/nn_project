import * as tf from '@tensorflow/tfjs';
// import cv from 'opencv';
const { Utils } = window;
const utils = new Utils('errorMessage');

const FPS = 10;
const video = document.getElementById('videoInput');
const predEmotion = document.getElementById('emotion');
const predProb = document.getElementById('prob');
const indexToEmotion =
  ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];

run();

function run() {
  const onVideoStarted = async () => {
    video.width = video.videoWidth;
    video.height = video.videoHeight;

    await runRecognizer();
  };

  utils.loadOpenCv(() => {
    const faceCascadeFile = 'haarcascade_frontalface_default.xml';
    utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
        console.log(`${faceCascadeFile} loaded`);
        utils.startCamera('qvga', onVideoStarted, 'videoInput');
    });
  });
}

async function runRecognizer() {
  const { cv } = window;
  const model = await loadModel();

  const src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  const dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  const gray = new cv.Mat();
  const cap = new cv.VideoCapture(video);
  const faces = new cv.RectVector();
  const classifier = new cv.CascadeClassifier();

  classifier.load('haarcascade_frontalface_default.xml');

  async function processVideo() {
    try {
      const begin = Date.now();
      cap.read(src);
      src.copyTo(dst);
      cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);

      // Detect faces
      classifier.detectMultiScale(gray, faces, 1.1, 3, 0);

      // Draw faces
      for (let i = 0; i < faces.size(); ++i) {
        const face = faces.get(i);
        const point1 = new cv.Point(face.x, face.y);
        const point2 = new cv.Point(face.x + face.width, face.y + face.height);
        cv.rectangle(dst, point1, point2, [255, 0, 0, 255]);
      }

      if (faces.size() > 0) {
        const face = faces.get(0);
        const faceRect = new cv.Rect(
          face.x,
          face.y,
          face.width,
          face.height,
        );
        const size = new cv.Size(48, 48);
        const extractedFace = src.roi(faceRect);
        const greyFace = new cv.Mat();
        cv.cvtColor(extractedFace, greyFace, cv.COLOR_RGBA2GRAY, 0);
        cv.resize(greyFace, greyFace, size);

        const input = tf.tensor1d(greyFace.data).reshape([1, 48, 48, 1]);
        const pred = model.predict(input);

        const idx = await pred.argMax(1).array();
        const prob = await pred.max(1).array();
        const emotion = indexToEmotion[idx[0]];

        predEmotion.innerHTML = emotion;
        predProb.innerHTML = prob;

        cv.imshow('ebalo', greyFace);
      }

      cv.imshow('canvasOutput', dst);
      let delay = 1000 / FPS - (Date.now() - begin);
      setTimeout(processVideo, delay);
    } catch (error) {
      utils.printError(error);
    }
  }
  setTimeout(processVideo, 0);
}

async function loadModel({
  path = '/models/vgg-based/model.json',
  summary = false
} = {}) {
  try {
    const model = await tf.loadLayersModel(path);

    if (summary) console.log(model.summary());
    return model;
  } catch (error) {
    console.error(`Error loading model ${path}\n${error}`);
  }
}
