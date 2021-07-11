"""
Object detector based on Model API
"""

import cv2
import numpy as np
from openvino.inference_engine import IECore
import sys
import logging as log
import argparse
import pathlib
from time import perf_counter

sys.path.append('C:\\Program Files (x86)\\Intel\\openvino_2021.4.582\\deployment_tools\\open_model_zoo\\demos\\common\\python')
import models
from pipelines import AsyncPipeline
from images_capture import open_images_capture


def get_plugin_configs(device, num_streams, num_threads):
    config_user_specified = {}

    devices_nstreams = {}
    if num_streams:
        devices_nstreams = {device: num_streams for device in ['CPU', 'GPU'] if device in device} \
            if num_streams.isdigit() \
            else dict(device.split(':', 1) for device in num_streams.split(','))

    if 'CPU' in device:
        if num_threads is not None:
            config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
        if 'CPU' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                if int(devices_nstreams['CPU']) > 0 \
                else 'CPU_THROUGHPUT_AUTO'

    if 'GPU' in device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                if int(devices_nstreams['GPU']) > 0 \
                else 'GPU_THROUGHPUT_AUTO'

    return config_user_specified


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an .xml \
        file with a trained model.', required=True, type=str)
    parser.add_argument('-i', '--input', help='Path to \
        image file', required=True, type=str)
    parser.add_argument('-d', '--device', help='Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)', default='CPU', type=str)
    parser.add_argument('-c', '--classes', help='File containing classes \
        names', type=str, default=None)
    parser.add_argument('-t', '--prob_threshold', default=0.8, type=float,
        help='Optional. Probability threshold for detections filtering.')
    return parser
  
  
def draw_detections(frame, detections, labels, threshold):
    with open(labels, 'r') as f:
        classes_list = f.read().split('\n')
    label_map = dict(enumerate(classes_list))
    size = frame.shape[:2]
    for detection in detections:
        score = detection.score

        # If score more than threshold, draw rectangle on the frame
        if score > threshold:
            cv2.rectangle(frame, (int(detection.xmin), int(detection.ymin)), (int(detection.xmax), int(detection.ymax)), (0, 255, 0), 1)
            cv2.putText(frame, f"Object {label_map[detection.id]} with {detection.score:.2f}", (int(detection.xmin), int(detection.ymin - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 1)

    return frame


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info("Start OpenVINO object detection")

    # Initialize data input
    cap = open_images_capture(args.input, True)

    # Initialize OpenVINO
    ie = IECore()

    # Initialize Plugin configs
    plugin_configs = get_plugin_configs('CPU', 0, 0)

    # Load YOLOv3 model
    detector = models.YOLO(ie, pathlib.Path(args.model), labels=args.classes, threshold=args.prob_threshold,
                           keep_aspect_ratio=True)

    # Initialize async pipeline
    detector_pipeline = AsyncPipeline(ie, detector, plugin_configs, device='CPU', max_num_requests=1)

    # Инициализировать объект записи видео
    output = cv2.VideoWriter('output_camera.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                             (1280, 720))

    counter = 1
    while True:

        # Get one image
        img = cap.read()

        start_time = perf_counter()

        # Start processing frame asynchronously
        frame_id = 0
        detector_pipeline.submit_data(img, frame_id, {'frame': img, 'start_time': 0})

        # Wait for processing finished
        detector_pipeline.await_any()
        
        # Get detection result
        results, meta = detector_pipeline.get_result(frame_id)
    
        # Draw detections in the image
        draw_detections(img, results, args.classes, args.prob_threshold)

        # Show image
        cv2.imshow('Image with detections', img)
        output.write(img)

        end_time = perf_counter()

        log.info(f" frame {counter} was processed with {end_time - start_time} seconds")
        counter += 1

        # Wait 1 ms and check pressed button to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # Destroy all windows
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    sys.exit(main())
