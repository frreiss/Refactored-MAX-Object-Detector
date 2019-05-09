# import tensorflow as tf
import tensorflowjs as tfjs
from tensorflowjs.converters import tf_saved_model_conversion

saved_model_path = "./saved_model_js"
output_path = "tfjs_models"

def convert_to_tfjs(input_dir, output_dir):
    output_node_names = "detection_boxes,detection_classes,detection_scores,num_detections"

    tf_saved_model_conversion.convert_tf_saved_model(input_dir, 
        output_node_names,
        output_dir,
        skip_op_check=True)

def main():
    # Reference: https://github.com/tensorflow/tfjs-converter/blob/0.8.x/python/tensorflowjs/converters/converter.py
    convert_to_tfjs(saved_model_path, output_path)


if __name__ == "__main__":
    main()