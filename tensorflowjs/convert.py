# import tensorflow as tf
import tensorflowjs as tfjs
from tensorflowjs.converters import tf_saved_model_conversion_pb

saved_model_path = "../saved_model_js"
output_path = "web_model"

def convert_to_tfjs(input_dir, output_dir):
    # Reference: https://github.com/tensorflow/tfjs-converter/blob/0.8.x/python/tensorflowjs/converters/converter.py
    output_node_names = "detection_boxes,detection_classes,detection_scores,num_detections"

    tf_saved_model_conversion_pb.convert_tf_saved_model(input_dir, 
        output_node_names,
        output_dir,
        saved_model_tags='serve',
        quantization_dtype=None,
        skip_op_check=False,
        strip_debug_ops=True)

def main():
    convert_to_tfjs(saved_model_path, output_path)


if __name__ == "__main__":
    main()