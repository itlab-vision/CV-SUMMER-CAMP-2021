"""
Classification sample

Command line to run:
python ie_classification_sample.py -i image.jpg \
    -m resnet-50.xml -w resnet-50.bin -c imagenet_synset_words.txt
"""

import os
import cv2
import sys
import argparse
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore


class InferenceEngineClassifier:
    def __init__(self, configPath=None, weightsPath=None,
            device='CPU', extension=None, classesPath=None):

        # Add code for Inference Engine initialization
        self.ie = IECore()
        # Add code for model loading
        self.net = self.ie.read_network(model=configPath)
        self.exec_net = self.ie.load_network(network=self.net, device_name=device)
        # Add code for classes names loading
        if classesPath is None:
            self.classes = [(i, '') for i in range(1000)]
        else:
            with open(classesPath, 'r') as fp:
                self.classes = [tuple(line.rstrip().split(maxsplit=1)) for line in fp]

        return

    def get_top(self, images_probs, topN=1):
        # Assoisiate probabilities with class names
        result = [dict() for i in range(len(images_probs))]
        for i, image_probs in enumerate(images_probs):
            for j, prob in enumerate(image_probs):
                for k in range(len(prob)):
                    result[i][self.classes[k][0]] = prob[k][0][0]

        # Return top entries based on highest probabilities
        return [sorted(result[i].items(), key=lambda i: i[1], reverse=True)[:topN] for i in range(len(result))]

    def _prepare_image(self, image, h, w):

        # Add code for image preprocessing
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        return image

    def classify(self, images):
        # Get data about input and output from neural network
        input_blob = next(iter(self.net.input_info))
        output_blob = next(iter(self.net.outputs))

        # Get required input shape for input
        n, c, h, w = self.net.inputs[input_blob].shape

        images_new = [np.ndarray(shape=(n, c, h, w)) for i in range(len(images))]
        for i, image in enumerate(images):
            images_new[i][0] = self._prepare_image(image, h, w)

        # Classify the images and get result tensors
        output = [self.exec_net.infer(inputs={input_blob: images_new[i]}) for i in range(len(images))]
        probabilities = [output[i][output_blob] for i in range(len(images))]

        return probabilities


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an .xml \
        file with a trained model.', required=True, type=str)
    parser.add_argument('-w', '--weights', help='Path to an .bin file \
        with a trained weights.', required=True, type=str)
    parser.add_argument('-i', '--input', help='Path to \
        image file', required=True, type=str)
    parser.add_argument('-l', '--cpu_extension', help='MKLDNN \
        (CPU)-targeted custom layers.Absolute path to a shared library \
        with the kernels implementation', type=str, default=None)
    parser.add_argument('-d', '--device', help='Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)', default='CPU', type=str)
    parser.add_argument('-c', '--classes', help='File containing classes \
        names', type=str, default=None)
    return parser


def get_images(image_dir_path):
    return os.listdir(os.path.join(*image_dir_path.split("/")))


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
        level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Start IE classification sample")

    # Create InferenceEngineClassifier object
    ie_classifier = InferenceEngineClassifier(configPath=args.model,
                                              weightsPath=args.weights, device=args.device,
                                              extension=args.cpu_extension, classesPath=args.classes)
    image_names = get_images(args.input)
    # Read images
    images = [cv2.imread(os.path.join(*args.input.split("/"), image_name)) for image_name in image_names]
    # Classify images
    probs = ie_classifier.classify(images)
    # Get top 5 predictions for each
    predictions = ie_classifier.get_top(probs, 5)
    # Show
    for i, image_name in enumerate(image_names):
        log.info(f"Predictions for {image_name}: " + str(predictions[i]))

    return


if __name__ == '__main__':
    sys.exit(main())
