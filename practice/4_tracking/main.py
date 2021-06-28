import argparse
import logging as log
from pathlib import Path
import sys
from tqdm import tqdm

from common.annotation import AnnotationStorage, emulate_reallife_detector
from tracker import Tracker, convert_tracks_to_annotation_storage
from demonstrator import Demonstrator
from evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Simple object tracker demo")
    parser.add_argument("--annotation", required=True, help="Path to the annotation file")
    parser.add_argument("--images_folder", help="Path to the folder with images")
    parser.add_argument("--dst_folder", required=True, help="Path to the destination folder")
    parser.add_argument("--verbose", action="store_true", help="If debug messages should be logged")
    parser.add_argument("--max_frame_index", help="If it is set, it points max frame_index to handle")
    parser.add_argument("--probability_miss_detection", type=float, default=0.6,
                        help="Probability to miss a bounding box by detection, intended "
                             "for emulating reallife detector issues")
    parser.add_argument("--max_mistake_in_bbox_corners", type=int, default=3,
                        help="Possible mistake in bbox corner in detection, intended "
                             "for emulating reallife detector issues")
    parser.add_argument("--max_output_image_width", type=int, default=1080,
                        help="Max image width for diplaying/saveing output images")
    parser.add_argument("--show", action="store_true",
                        help="Add this flag to make demo show the tracking result using OpenCV imshow function")
    args = parser.parse_args()

    if args.verbose:
        log_level = log.DEBUG
    else:
        log_level = log.INFO
    log.basicConfig(format="%(levelname)s|%(filename)s:%(lineno)d|%(funcName)s: %(message)s",
                    level=log_level, stream=sys.stdout)

    dst_folder = Path(args.dst_folder)
    dst_folder.mkdir(exist_ok=True)

    max_frame_index = args.max_frame_index
    if max_frame_index is not None:
        max_frame_index = int(max_frame_index)
    else:
        max_frame_index = sys.maxsize

    log.info("Begin reading annotation from the file '{}' and emulating reallife detector issues".format(args.annotation))
    groundtruth_annotation_storage = AnnotationStorage(args.annotation)
    input_annotation_storage = emulate_reallife_detector(groundtruth_annotation_storage,
                                                         probability_miss_detection=args.probability_miss_detection,
                                                         max_mistake_in_bbox_corners=args.max_mistake_in_bbox_corners)
    log.info("End reading annotation from the file '{}' and emulating reallife detector issues".format(args.annotation))

    demonstrator = Demonstrator(args.images_folder, dst_folder, args.max_output_image_width, args.show)
    evaluator = Evaluator(groundtruth_annotation_storage, max_frame_index)

    tracker = Tracker(num_frames_to_remove_track=30,
                      num_objects_to_make_track_valid=5,
                      affinity_threshold=0.8)

    log.info("Begin tracking")
    list_frame_indexes = input_annotation_storage.get_list_of_frame_indexes()
    list_frame_indexes = [frame_index for frame_index in list_frame_indexes if frame_index <= max_frame_index]
    for frame_index in tqdm(list_frame_indexes, desc="Tracking"):
        if frame_index > max_frame_index:
            break
        log.debug("begin handling frame index={}".format(frame_index))
        det_objs = input_annotation_storage.get_detected_objects_for_frame_index(frame_index)
        tracker.add_objects(det_objs)
        log.debug("end handling frame index={}".format(frame_index))
    log.info("End tracking")

    log.info("Begin converting tracks and writing to file")
    valid_tracks = tracker.get_all_valid_tracks()
    result_annotation_storage = convert_tracks_to_annotation_storage(valid_tracks)

    result_file_path = dst_folder / "result.txt"
    result_annotation_storage.write_to_file(result_file_path)
    log.info("End converting tracks and writing to file")

    log.info("Begin evaluation")
    res_eval_path = dst_folder / "evaluation.txt"
    num_hits, num_misses, recall = evaluator.evaluate(result_annotation_storage, res_eval_path)
    log.info("Num groundtruth tracks that were found: num_hits={}".format(num_hits))
    log.info("Num groundtruth tracks that were missed: num_misses={}".format(num_misses))
    log.info("Total quality: recall={:.2f}".format(recall))

    log.info("End evaluation, the result is written to the file {}".format(res_eval_path))

    log.info("Begin writing demo to images in the folder {}".format(dst_folder))
    demonstrator.make_demonstration(result_annotation_storage)
    log.info("End writing demo to images in the folder {}".format(dst_folder))
    log.info("Done")

if __name__ == "__main__":
    main()
