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

### Part 1: Generate the Graph

Since the core model is implemented in TensorFlow, the graph is represented as a SavedModel "file". The script that generates this file is at `build_graph.py`, and its output goes to `outputs/saved_model`.

Commands to copy and paste:
```
env/bin/python ./build_graph.py
```


### Part 2: Generate the Deployment Artifact

**TODO steps for various output targets**

