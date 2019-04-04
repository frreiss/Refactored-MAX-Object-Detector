# Refactored-MAX-Object-Detector

Prototype of new structure for the existing project 
[IBM/MAX-Object-Detector](https://github.com/IBM/MAX-Object-Detector).

## General Design Principles

* The input to the model construction is the **trained graph**, i.e. the large binary files of graphs and weights that the training script produces at the completion of the training run.
* Each version of the trained graph will be an immutable, read-only resource that we will store indefinitely. We will store these versions of the trained graph on **[TBD]**.
* We will *not* directly deploy the trained graph. Instead, we will use the trained graph as an input to construct a standardized **inference graph**.
* We will embed as much inference functionality as possible into the inference graph. This includes preprocessing operations such as parsing the JPEG file format into an in-memory bitmap tensor; as well as postprocessing operations like mapping class IDs to class names.
* For operations that absolutely cannot be implemented inside the inference graph, we will write a **preprocessing function** and a **postprocessing function**. These functions will take (binary, parsed) JSON as an input format and produce (binary) JSON as an output format. We will implement the functions in both Python and JavaScript/TypeScript.
* We will build utilities that use the inference graph plus pre/postprocessing functions to produce deployable artifacts for a variety of inference platforms. These utilities will work for all MAX models.
* We will preprocess the graph as much as possible to remove portions that are not necessary for inference.
* We will represent the inference graph in one of the following standard formats:
	* TensorFlow models: SavedModel file containing a frozen graph, with a single signature under the `tag_constants.SERVING` key.
	* Keras models: Format TBD. We will either use a `.h5` file ; or we will convert to `tf.keras` and use a `SavedModel` file consistent with the format for TensorFlow models.
	* Pytorch models: ONNX graph. Exact format still TBD.


## Contents of this repo:

**TODO: Describe directory structure once it settles down**


## Compiling the model

Model compilation happens in two phases: First, generate a graph suitable for inference; then wrap the graph in an appropriate container for the target deployment environment.


### Part 0: Prepare an Environment

Install Anaconda on your local machine, then use the script `env.sh` to create and populate local Anaconda environment in the directory `./env`.

Commands to copy and paste:
```
./env.sh
conda activate ./env
```

### Part 1: Generate the graph

Since the core model is implemented in TensorFlow, the graph is represented as a SavedModel "file". The script that generates this file is at `build_graph.py`, and its output goes to `outputs/saved_model`.

Commands to copy and paste:
```
env/bin/python ./build_graph.py
```

The resulting graph goes to a TensorFlow SavedModel located at `[project root]/saved_model`.

### Part 2: Test the graph locally

The script `test_local.py` instantiates the model graph locally, sends an example image through the graph, and prints the result. Commands to copy and paste:
```
env/bin/python ./test_local.py
```
The output should look something like this:
```
[...]
Predictions: [{'label': 'bear', 'probability': 0.991337776184082, 'detection_box': [0.27232617139816284, 0.24801963567733765, 0.8350169062614441, 0.7980572581291199]}]
Result:
{
    "status": "ok",
    "predictions": [
        {
            "label": "bear",
            "probability": 0.991337776184082,
            "detection_box": [
                0.27232617139816284,
                0.24801963567733765,
                0.8350169062614441,
                0.7980572581291199
            ]
        }
    ]
}
```

### Part 3: Deploy the model to Watson Machine Learning

Start by performing the following manual steps:
  * Create a file `ibm_cloud_credentials.json` in this directory, if such a
    file doesn't already exist.
    Initialize the file with an empty JSON record, i.e. "{ }".
  * Create a Watson Machine Learning (WML) instance.
  * Navigate to your WML instance's web UI and click on the "Service
    credentials" link, then click on "New credential" to create a new set of
    service credentials. Copy the credentials into `ibm_cloud_credentials.json`
    under the key "WML_credentials".
    
If you are running on a shared machine, make sure that other user IDs cannot read `ibm_cloud_credentials.json`. **DO NOT CHECK CREDENTIALS INTO GITHUB.**

Then run the script `deploy_wml.py`:
```
env/bin/python deploy_wml.py
```
When the script finishes, it prints out three additional lines to add to `ibm_cloud_credentials.json`; something like:
```
[...]
Lines to add to ibm_cloud_credentials.json:
    "WML_model_ID": "[long hexadecimal string]",
    "WML_deployment_ID": "[long hexadecimal string]",
    "WML_function_url": "https://us-south.ml.cloud.ibm.com/v3/wml_instances/[long hexadecimal string]/deployments/[long hexadecimal string]/online"
```
Add those lines of JSON to `ibm_cloud_credentials.json`, replacing any previous values of those lines.

### Part 4: Test the deployed model on Watson Machine Learning with pre- and post-processing code running locally

The script `test_wml_local.py` does the following steps:
1. Read in an example image
2. Run preprocessing code for scoring on WML locally on your laptop
3. Submit the preprocessed inference request to the deployed model on WML
4. Run some postprocessing code to reformat the response
5. Print the response

To run the script, use the following command:
```
env/bin/python test_wml_local.py
```
*This script is not currently working, but it should be working soon*

### Part 5: Test the deployed model on Watson Machine Learning with pre- and post-processing code running on Watson Machine Learning

The script `test_wml.py` uses a the WML function deployed earlier by `deploy_wml.py` to perform pre- and post-processing operations. The script issues a REST request against the deployed function and prints out the response. To run the script, use the following command:
```
env/bin/python test_wml.py
```
*This script is not currently working, but it should be working soon*

**TODO steps for additional output targets**

