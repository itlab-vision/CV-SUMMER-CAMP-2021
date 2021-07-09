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
        self.net = self.ie.read_network(model=configPath, weights=weightsPath)
        self.classes = classesPath
        
        # Add code for model loading
        self.exec_net = self.ie.load_network(network=self.net, device_name=device, num_requests=2)

        if extension:
            self.ie.add_extension(unicode_extension_path=extension, unicode_device_name=device)
        
        return

    def get_top(self, prob, topN=1):
        result = np.ndarray(shape=(prob.shape[0], topN))
        for i in range(len(prob)):
        # Add code for getting top predictions
            prob[i] = np.squeeze(prob[i])
            result[i] = np.argsort(prob[i])[-topN:][::-1]

        return result

    def _prepare_image(self, image, image_name, h, w):
    
        # Add code for image preprocessing
        if image.shape[:-1] != (h, w):
            log.warning("Image {} is resized from {} to {}".format(image_name, image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW

        return image

    def classify(self, images_):
        sizes = 224 if self.net.name == 'resnet-50-tf' else 227
        
        # Add code for image classification using Inference Engine
        input_blob = next(iter(self.net.input_info))
        out_blob = next(iter(self.net.outputs))

        n, c, h, w = self.net.input_info[input_blob].input_data.shape
        log.info(f"Batch size: {n}")
        log.info(f"Colors count: {c}")
        log.info(f"Height {h}")
        log.info(f"Width: {w}")

        images = np.ndarray(shape=(n, c, h, w))
        for i in range(n):
            image = cv2.imread(images_[i])
            image = self._prepare_image(image, images_[i], sizes, sizes)
            images[i] = image

        output = self.exec_net.infer(inputs={input_blob: images})
        output = output[out_blob]

        log.info("Classes count in model: " + str(max(self.net.outputs[out_blob].shape) - 1))

        return output


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an .xml \
        file with a trained model.', required=True, type=str)
    parser.add_argument('-w', '--weights', help='Path to an .bin file \
        with a trained weights.', required=True, type=str)
    parser.add_argument('-i', '--input', nargs='+', help='Path to \
         image file or files, if more than 1, should be separated \
        with comma without space or only with space; \
         example: a.jpg,b.jpg or a.jpg b.jpg', required=True, type=str)
    parser.add_argument('-l', '--cpu_extension', help='MKLDNN \
        (CPU)-targeted custom layers.Absolute path to a shared library \
        with the kernels implementation', type=str, default=None)
    parser.add_argument('-d', '--device', help='Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)', default='CPU', type=str)
    parser.add_argument('-c', '--classes', help='File containing classes \
        names', type=str, default=None)
    parser.add_argument('-n', '--number_predictions',
                        help='Amount of result predictions, 5 by default', type=int, default=5)
    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    #   more images processing
    images_data = []
    for i in range(len(args.input)):
        for name in args.input[i].split(','):
            if name != args.input[i]:
                images_data.append(name)

    if len(images_data) == 0:
        images_data = args.input

    # Create InferenceEngineClassifier object
    classifier = InferenceEngineClassifier(configPath=args.model, weightsPath=args.weights, device=args.device,
                                           extension=args.cpu_extension, classesPath=args.classes)

    classifier.net.batch_size = len(images_data)

    if len(images_data) > 1:
        log.info(f"Start IE classification sample on pictures {args.input} with model {args.model[:-4]}")
    else:
        log.info(f"Start IE classification sample on picture {args.input} with model {args.model[:-4]}")

    # Read image later
    #image = cv2.imread(name)

    # Classify batch of images
    prediction = classifier.classify(images_data)

    # Get top 5 predictions for each image
    top_size = args.number_predictions
    only_top = classifier.get_top(prediction, top_size)

    with open(args.classes, 'r') as f:
        count = sum(1 for _ in f)
    log.info(f"Classes count in file {args.classes}: " + str(count))

    names = []
    with open(args.classes, 'r') as f:
        data = f.read()
    if args.classes:
        for i in range(prediction.shape[0]):
            names.append([])
            for prediction_ind in only_top[i]:
                for ind, line in enumerate(data.split('\n')):
                    if prediction_ind == ind:
                        names[i].append(f"{ind+1} {line} {round(prediction[i][ind] * 100, 2)}%")
                        break

    log.info(f'Top {top_size} predictions:')  # + str(only_top)
    counter = 1
    old_image_answer = []
    for image_answer in names:
        if image_answer == old_image_answer:
            break
        log.info(f'Predictions on {counter} image:')
        for answer in image_answer:
            log.info(answer)
        counter += 1
        log.info("")
        old_image_answer = image_answer

    return


if __name__ == '__main__':
    sys.exit(main())
