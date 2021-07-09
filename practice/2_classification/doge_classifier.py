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
    _device = 'CPU'

    def __init__(self, configPath=None, weightsPath=None,
            device='CPU', extension=None, classesPath=None):
        
        # Add code for Inference Engine initialization
        self.ie = IECore()
        # Add code for model loading
        self.net = self.ie.read_network(model=configPath)
        
        self._device = device
        
        return

    def get_top(self, prob, topN=1):
        results = []
        
        # Add code for getting top predictions
        for result in prob:
            result = np.squeeze(result)
            top_classes = np.argsort(result)[-topN :][::-1]
            top_probs = [result[x] for x in top_classes]
            results.append((top_classes, top_probs))
        
        return results

    def prepare_input(self, input):

        # Read images
        imgs_path = []
        is_correct = True
        input = input.split(',')
        if os.path.exists(input[0]):
            if os.path.isdir(input[0]):
                path = os.path.abspath(input[0])
                imgs_path = [os.path.join(path, file) for file in os.listdir(path)]
            elif os.path.isfile(input[0]):
                for image in input:
                    if not os.path.isfile(image):
                        raise ValueError('Incorrect input!')
                    imgs_path.append(os.path.abspath(image))
            else:
                raise ValueError('Incorrect input!')
        
        # Add code for image preprocessing
        input_blob = next(iter(self.net.inputs))
        n, c, h, w = self.net.inputs[input_blob].shape
        images = []
        for path in imgs_path:
            image = cv2.imread(path)
            image = cv2.resize(image, (h, w))
            images.append(image)
        images = np.array(images)
        images = images.transpose((0, 3, 1, 2))
        return images

    def classify(self, images):
        
        # Add code for image classification using Inference Engine
        out_blob = next(iter(self.net.outputs))
        input_blob = next(iter(self.net.inputs))

        self.net.batch_size = len(images)
        self.exec_net = self.ie.load_network(network=self.net, device_name=self._device,
            config={"DYN_BATCH_ENABLED": "YES"})

        output = self.exec_net.infer(inputs = {input_blob: images})

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
    ie_classifier = InferenceEngineClassifier(configPath=args.model, 
        weightsPath=args.weights, device='GPU', 
        extension=args.cpu_extension, classesPath=args.classes)

    # Read image
    images = ie_classifier.prepare_input(args.input)

    # Classify image
    prob = ie_classifier.classify(images)
    
    # Get top 5 predictions
    results = ie_classifier.get_top(prob, 5)
    
    # print result
    with open(args.classes, 'r') as f:
        classes = [line.strip('\n') for line in f]
        for i, result in enumerate(results):
            top_classes = result[0]
            top_probs = result[1]
            log.info("Predictions for image {}:\n".format(i) + 
                '\n'.join([classes[idx][10 :] + ' ' + str(top_probs[j]) for j, idx in enumerate(top_classes)]))
    
    return


if __name__ == '__main__':
    sys.exit(main())
