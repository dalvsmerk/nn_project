import * as tf from '@tensorflow/tfjs';
// import cv from 'opencv';
const { Utils } = window;
const utils = new Utils('errorMessage');

const FPS = 15;
const video = document.getElementById('videoInput');
const indexToEmotion =
  ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];

const onVideoStarted = async () => {
  const scale = 1.5;
  video.width = video.videoWidth * scale;
  video.height = video.videoHeight * scale;

  await runRecognizer();
};

utils.loadOpenCv(() => {
  const faceCascadeFile = 'haarcascade_frontalface_default.xml';
  utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
      console.log(`${faceCascadeFile} loaded`);
      utils.startCamera('qvga', onVideoStarted, 'videoInput');
  });
});

/**
 * Runs the prediction loop from the video buffer
 */
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
  hideSpinner();

  async function processVideo() {
    try {
      const begin = Date.now();
      cap.read(src);
      src.copyTo(dst);
      cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);

      // Detect faces
      classifier.detectMultiScale(gray, faces, 1.1, 3, 0);

      const inputFaces = extractGreyscaleFaces(src, faces);
      const predictions = await predictBatch(model, inputFaces);
      drawPredictions(faces, predictions, dst);

      cv.imshow('canvasOutput', dst);

      let delay = 1000 / FPS - (Date.now() - begin);
      setTimeout(processVideo, delay);
    } catch (error) {
      console.error(error);
    }
  }
  setTimeout(processVideo, 0);
}

/**
 * Transform extracted faces to greyscale model input
 *
 * @param {cv.Mat} src Buffer to extract faces from
 *
 * @returns {tf.Tensor4D} Tensor of input faces with shape [BATCH_SIZE, 48, 48, 1]
 */
function extractGreyscaleFaces(src, faces) {
  const detectedFaces = faces.size();
  const inputFaces = [];

  for (let i = 0; i < detectedFaces; i++) {
    const face = faces.get(i);
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

    inputFaces.push(greyFace.data);
  }

  return tf.tensor(inputFaces).reshape([detectedFaces, 48, 48, 1]);
}

/**
 * Predicts facial expression labels for a batch of faces
 *
 * @param {tf.LayersModel} model Facial expression classifier
 * @param {tf.Tensor4D} faces Batch of extracted greyscale faces
 *
 * @returns {array} Array of prediction [[label, probability]]
 */
async function predictBatch(model, faces) {
  if (!faces.shape[0]) return [];

  const predictions = model.predict(faces);

  const labelIdx = await predictions.argMax(1, 1).array();
  const labelProb = await predictions.max(1).array();

  const composeLabelsProbs = (acc, idx, i) =>
    [...acc, [indexToEmotion[idx], labelProb[i]]];

  return labelIdx.reduce(composeLabelsProbs, []);
}

/**
 * Draws face boxes, predicted expression labels and probabilities
 *
 * @param {cv.RectVector} faces Boundary boxes of detected faces
 * @param {array} preds Predictions for detected faces [[label, probability]]
 * @param {cv.Mat} dst Where to display the predictions
 */
function drawPredictions(faces, preds, dst) {
  if (faces.size() != preds.length) {
    throw new Error('Size of faces doesn\'t correspond to size of predictions');
  }

  for (let i = 0; i < faces.size(); i++) {
    const face = faces.get(i);
    const point1 = new cv.Point(face.x, face.y);
    const point2 = new cv.Point(face.x + face.width, face.y + face.height);

    const facePrediction = preds[i];
    const singleProb = facePrediction[1].toFixed(2);
    const emotion = facePrediction[0];
    const fontFace = cv.FONT_HERSHEY_PLAIN;
    const fontScale = 1;
    const text = `${emotion} ${singleProb}`;
    const textPosition = new cv.Point(face.x, face.y - 5);
    const textColor = new cv.Scalar(0, 255, 0, 255);

    cv.putText(dst, text, textPosition, fontFace, fontScale, textColor);
    cv.rectangle(dst, point1, point2, [0, 255, 0, 255]);
  }
}

/**
 * Loads pre-trained model graph and weights
 *
 * @param {bool} summary Controls visibility of model summary
 *
 * @returns {tf.LayersModel} Classifier model
 */
async function loadModel(summary = false) {
  try {
    const model = await tf.loadLayersModel('/models/vgg-based/model.json');

    if (summary) console.log(model.summary());
    return model;
  } catch (error) {
    console.error(`Error loading model ${path}\n${error}`);
  }
}

function hideSpinner() {
  const spinner = document.querySelector('.spinner-container');
  spinner.classList.add('spinner-container--hidden');
}
