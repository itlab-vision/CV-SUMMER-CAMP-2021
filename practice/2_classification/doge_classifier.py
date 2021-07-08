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
    def __init__(
        self, configPath=None, weightsPath=None,
        device='CPU', extension=None, classesPath=None
    ):
        # Inference Engine initialization
        ie = IECore()

        # Model loading
        self.network = ie.read_network(configPath, weightsPath)
        self.exec_net = ie.load_network(self.network, device)

        # Load class names from file
        with open(classesPath, 'r') as fp:
            self.classes = [tuple(line.rstrip().split(maxsplit=1)) for line in fp]

        return

    def get_top(self, probs, topN=1):
        # Assoisiate probabilities with class names
        result = {}
        for i, prob in enumerate(probs):
            result[self.classes[i]] = prob[0][0]

        # Return top entries based on highest probabilities
        return sorted(result.items(), key=lambda i: i[1], reverse=True)[:topN]

    def _prepare_image(self, image, h, w):

        # Resize image
        old_size = image.shape[::-1][:2]
        new_size = (w, h)
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        log.warning(f'Image was resized from {old_size} to {new_size}')

        # Convert from RGBRGBRGB to RRRGGGBBB
        image = image.transpose((2, 0, 1))

        return image

    def classify(self, image):
        probabilities = None

        # Get data about input and output from neural network
        input_blob = next(iter(self.network.input_info))
        output_blob = next(iter(self.network.outputs))

        # Get required input shape for input
        n, c, h, w = self.network.inputs[input_blob].shape

        images = np.ndarray(shape=(n, c, h, w))
        images[0] = self._prepare_image(image, h, w)

        # Classify the image and get result tensor
        output = self.exec_net.infer(inputs = {input_blob: images})
        probabilities = output[output_blob][0]

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


def main():
    log.basicConfig(
        format="[ %(levelname)s ] %(message)s",
        level=log.INFO, stream=sys.stdout
    )
    args = build_argparser().parse_args()

    log.info("Start IE classification sample")

    # Create InferenceEngineClassifier object
    ie_classifier = InferenceEngineClassifier(
        configPath=args.model,
        weightsPath=args.weights,
        device=args.device,
        extension=args.cpu_extension,
        classesPath=args.classes,
    )

    # Read image
    image = cv2.imread(args.input)

    # Classify image
    probs = ie_classifier.classify(image)

    # Get top 5 predictions
    preds = ie_classifier.get_top(probs, 5)
    for pred in preds:
        log.info(f'Predicted ({pred[1]*100:0>5.2f}%) [{pred[0][0]}] {pred[0][1]}')


if __name__ == '__main__':
    sys.exit(main())
