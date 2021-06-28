from copy import deepcopy
from pathlib import Path
import random
from common.common_objects import DetectedObject, validate_detected_object, Bbox

class AnnotationObject:
    def __init__(self, detect_obj, track_id):
        validate_detected_object(detect_obj)
        self.detect_obj = detect_obj
        self.track_id = track_id # the groundtruth track id

class AnnotationStorage:
    def __init__(self, annotation_file_path=None):
        self.annotation_file_path = annotation_file_path
        self.objects_by_frame = {}
        self.all_objects = []

        if annotation_file_path is None:
            return

        with Path(annotation_file_path).open() as f:
            for line in f:
                line = line.strip()
                chunks = line.split()
                assert 3 <= len(chunks) <= 4, "Error in line={}".format(line)
                frame_index = int(chunks[0])
                track_id = int(chunks[1])

                bbox_coords = chunks[2].split(",")
                assert len(bbox_coords) == 4, "Error in line={}".format(line)
                tl_x, tl_y, br_x, br_y = bbox_coords
                tl_x = int(tl_x)
                tl_y = int(tl_y)
                br_x = int(br_x)
                br_y = int(br_y)

                if len(chunks) == 4:
                    appearance_feature = chunks[3].split(",")
                    appearance_feature = [float(v) for v in appearance_feature]
                else:
                    appearance_feature = [1.]

                bb = Bbox(tl_x, tl_y, br_x, br_y)

                o = DetectedObject(frame_index, bb, appearance_feature)

                ann_obj = AnnotationObject(detect_obj=o, track_id=track_id)

                if frame_index not in self.objects_by_frame:
                    self.objects_by_frame[frame_index] = []

                self.objects_by_frame[frame_index].append(ann_obj)
                self.all_objects.append(ann_obj)

    def get_detected_objects_for_frame_index(self, frame_index):
        if frame_index not in self.objects_by_frame:
            return []
        res = []
        for ann_obj in self.objects_by_frame[frame_index]:
            o = ann_obj.detect_obj
            assert o.frame_index == frame_index
            res.append(o) #return DetectedObject without track id-s

        return res

    def get_list_of_frame_indexes(self):
        return sorted(list(self.objects_by_frame.keys()))

    @staticmethod
    def create_annotation_storage_from_list(annotation_objects):
        annotation_objects = deepcopy(annotation_objects)
        annotation_storage = AnnotationStorage()

        annotation_storage.all_objects = annotation_objects

        prev_frame_index = -1
        for ann_obj in annotation_objects:
            frame_index = ann_obj.detect_obj.frame_index
            assert prev_frame_index <= frame_index

            if frame_index not in annotation_storage.objects_by_frame:
                annotation_storage.objects_by_frame[frame_index] = []

            annotation_storage.objects_by_frame[frame_index].append(ann_obj)
            prev_frame_index = frame_index

        return annotation_storage

    def write_to_file(self, ann_file_path):
        with Path(ann_file_path).open("w") as f_dst:
            for ann_obj in self.all_objects:
                frame_index = ann_obj.detect_obj.frame_index
                track_id = ann_obj.track_id
                bbox = ann_obj.detect_obj.bbox
                appearance_feature = ann_obj.detect_obj.appearance_feature

                line = "{:06} {:04} {},{},{},{}".format(frame_index,
                                                  track_id,
                                                  bbox.tl_x, bbox.tl_y, bbox.br_x, bbox.br_y)
                if appearance_feature is not None and len(appearance_feature) > 0:
                    features_str = ",".join("{:.2f}".format(v) for v in appearance_feature)
                    line += " " + features_str

                f_dst.write(line + "\n")

def emulate_reallife_detector(annotation_storage,
                              probability_miss_detection=0.5,
                              max_mistake_in_bbox_corners=3):
    all_objects = deepcopy(annotation_storage.all_objects)
    res_all_objects = []
    for ann_obj in all_objects:
        miss_val = random.random()
        if miss_val <= probability_miss_detection:
            # we skip this annotation object, emulating miss of detector
            continue

        D = max_mistake_in_bbox_corners
        det_obj = ann_obj.detect_obj
        bbox = det_obj.bbox
        tl_x = bbox.tl_x + random.randint(-D, D)
        tl_y = bbox.tl_y + random.randint(-D, D)
        br_x = bbox.br_x + random.randint(-D, D)
        br_y = bbox.br_y + random.randint(-D, D)

        new_bbox = Bbox(tl_x, tl_y, br_x, br_y)
        new_det_obj = DetectedObject(det_obj.frame_index,
                                     new_bbox,
                                     det_obj.appearance_feature)

        ann_obj.detect_obj = new_det_obj
        res_all_objects.append(ann_obj)

    return AnnotationStorage.create_annotation_storage_from_list(res_all_objects)
