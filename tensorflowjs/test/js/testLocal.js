const tf = require('@tensorflow/tfjs');
const tfn = require('@tensorflow/tfjs-node');
const fs = require('fs');

const prepost = require("../../prePost");
const inferenceRequest = require("../../inferenceRequest");

// https://stackoverflow.com/questions/52665923/tfjs-node-how-to-load-model-from-url
global.fetch = require('node-fetch');

const modelUrl = 'file://web_model/tensorflowjs_model.pb';
const weightsUrl = 'file://web_model/weights_manifest.json';

const SAMPLE_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG"
 

async function loadModel(modelUrl, weightsUrl) {
    const model = await tf.loadFrozenModel(modelUrl, weightsUrl);
    // TFJS 1.0 call
    // const model = await tf.loadLayersModel(modelHandler);
    return model;
}


async function main() {
    const model = await loadModel(modelUrl, weightsUrl);
    // console.log(model.executor._outputs)
    
    let request = new inferenceRequest.InferenceRequest();
    let handler = new prepost.ObjectDetectorHandler();

    fetch(SAMPLE_IMAGE_URL)
        .then(function(response) {
            return response.arrayBuffer();
        }).then(function(buffer) {
             request.rawInputs['image'] = [new Uint8ClampedArray(buffer)];
             request.rawInputs['threshold'] = 0.7;
             try {
                handler.preprocess(request);
            } catch(err) {
                console.log(err);
            }
        }).then(function() {
            let inputTensors = tf.stack(request.processedInputs['image']);
            var result = model.executeAsync(inputTensors);
            return result;
        }).then(function(output) {
            request.rawOutputs["detectionScores"] = output[0].arraySync();
            request.rawOutputs["detectionBoxes"] = output[1].arraySync();
            request.rawOutputs["numDetections"] = output[2].arraySync();
            request.rawOutputs["detectionClasses"] = output[3].arraySync();
        }).then(function(){
            handler.postprocess(request);
            console.log(request.processedOutputs.predictions)
        })

}

main();