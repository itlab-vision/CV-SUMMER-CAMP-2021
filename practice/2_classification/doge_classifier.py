"""
Classification sample

Command line to run:
python ie_classification_sample.py -i image.jpg \
    -m resnet-50.xml -w resnet-50.bin -c imagenet_synset_words.txt
"""

# import os
# os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.3.394\\deployment_tools\\ngraph\\lib")
# os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.3.394\\deployment_tools\\inference_engine\\external\\tbb\\bin")
# os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.3.394\\deployment_tools\\inference_engine\\bin\\intel64\\Release")
# #os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.3.394\\deployment_tools\\inference_engine\\external\\hddl\\bin")
# os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.3.394\\opencv\\bin")

import os
import cv2
import sys
import argparse
import numpy as np
import logging as log
from numpy.core.fromnumeric import shape
from openvino.inference_engine import IENetwork, IECore


class InferenceEngineClassifier:
    def __init__(
        self,
        configPath=None,
        weightsPath=None,
        device="CPU",
        extension=None,
        classesPath=None,
        batch_size=1,
    ):

        # Add code for Inference Engine initialization
        self.ie = IECore()
        self.ie.set_config(config={"DYN_BATCH_ENABLED": "YES"}, device_name=device)

        # Add code for model loading
        self.net = self.ie.read_network(model=configPath)
        self.net.batch_size = batch_size

        self.exec_net = self.ie.load_network(network=self.net, device_name=device)

        # Add code for classes names loading
        self.labels = None

        if classesPath:
            with open(classesPath, "r") as f:
                self.labels = [line.split("'")[1].strip() for line in f]

        return

    def get_top(self, prob, topN=1):
        result = []

        # Add code for getting top predictions
        result = np.squeeze(prob)
        result = np.argsort(result)[-topN:][::-1]

        labels = [(" - " + self.labels[index] if self.labels else "") for index in result]

        return result, labels

    def _prepare_image(self, image, h, w):

        # Add code for image preprocessing
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))

        return image

    def classify(self, input):
        probabilities = None

        # Add code for image classification using Inference Engine
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))

        n, c, h, w = self.net.inputs[input_blob].shape

        images = np.zeros(shape=(len(input), c, h, w))

        for i, path in enumerate(input):
            image = cv2.imread(path)
            image = self._prepare_image(image, h, w)
            images[i] = image

        output = self.exec_net.infer(inputs={input_blob: images})
        probabilities = output[out_blob]

        return probabilities


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Path to an .xml \
        file with a trained model.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weights",
        help="Path to an .bin file \
        with a trained weights.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path to \
        image file",
        required=True,
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "-l",
        "--cpu_extension",
        help="MKLDNN \
        (CPU)-targeted custom layers.Absolute path to a shared library \
        with the kernels implementation",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)",
        default="CPU",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--classes",
        help="File containing classes \
        names",
        type=str,
        default=None,
    )
    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Start IE classification sample")

    # Create InferenceEngineClassifier object
    ie_classifier = InferenceEngineClassifier(
        configPath=args.model,
        weightsPath=args.weights,
        device=args.device,
        extension=args.cpu_extension,
        classesPath=args.classes,
        batch_size=len(args.input),
    )
    # Read image

    # Classify image
    prob = ie_classifier.classify(args.input)

    # Get top 5 predictions
    topN = 5
    
    for i, path in enumerate(args.input):
        print()

        log.warning(f"Image path: {path}")
        log.info(f"Top {topN}:")

        pred, labels = ie_classifier.get_top(prob[i], topN=topN)

        for index, label in zip(pred, labels):
            log.info("{}{}".format(index, label))

    return


if __name__ == "__main__":
    sys.exit(main())
