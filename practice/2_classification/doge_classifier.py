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
        #with open(args.labels, 'r') as f:
            #labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
        return

    def get_top(self, prob, topN=1):
        result = []
        
        result = np.squeeze(prob)
        result = np.argsort(result)[-topN:][::-1]
        # Add code for getting top predictions
        
        return result

    def _prepare_image(self, image, h, w):
    
        # Add code for image preprocessing
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        return image

    def classify(self, image):
        probabilities = None
        
        # Add code for image classification using Inference Engine
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))

        n, c, h, w = self.net.inputs[input_blob].shape

        image = self._prepare_image(image, h, w)

        output = self.exec_net.infer(inputs = {input_blob: image})

        output = output[out_blob]

        return output


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
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
        level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Start IE classification sample")

    # Create InferenceEngineClassifier object
    ie_classifier = InferenceEngineClassifier(
        configPath=args.model,
        weightsPath=args.weights, 
        device=args.device,
        extension='CPU', 
        classesPath=args.classes)

    # Read image
    img = cv2.imread(args.input)
        
    # Classify image
    prob = ie_classifier.classify(img)
    
    # Get top 5 predictions
    predictions = ie_classifier.get_top(prob, 5)
    
    # print result
    log.info("Predictions: " + str(predictions))

    return


if __name__ == '__main__':
    sys.exit(main())
