import numpy as np
import math
import logging as log
import sys
from tqdm import tqdm
from common.feature_distance import calc_features_similarity
from common.common_objects import DetectedObject, validate_detected_object, Bbox
from common.common_objects import get_bbox_center, get_dist, calc_bbox_area
from common.find_best_assignment import solve_assignment_problem
from common.annotation import AnnotationObject, AnnotationStorage

class Track:
    __next_track_id = 0
    def __init__(self, first_obj):
        self.objects = []
        self._track_id = Track.__next_track_id
        Track.__next_track_id += 1

        self.objects.append(first_obj)

    def _validate(self):
        assert len(self.objects) > 0
        for o in self.objects:
            validate_detected_object(o)
        for i in range(len(self.objects) - 1):
            self.objects[i].frame_index < self.objects[i+1].frame_index

    def add_object(self, o):
        self._validate()
        validate_detected_object(o)

        last_frame_index = self.objects[-1].frame_index
        if not last_frame_index < o.frame_index:
            raise RuntimeError("Add object={} to track with the last_frame_index={}".format(o, last_frame_index))

        self.objects.append(o)

    def last(self):
        return self.objects[-1]

    def get_id(self):
        return self._track_id

    def get_bbox_for_frame(self, cur_frame_ind):
        """Finds bbox for frame index using linear approximation"""
        self._validate()
        i_found = None
        for i, o in enumerate(self.objects):
            if o.frame_index == cur_frame_ind:
                return o.bbox
            if o.frame_index > cur_frame_ind:
                i_found = i
                break
        if i_found is None: # cur_frame_ind after the last frame_index in track
            return None

        if i_found == 0: # cur_frame_ind before the first frame_index in track
            return None
        log.debug("using linear approximation for track id={}, frame_index={}".format(self._track_id, cur_frame_ind))
        o1 = self.objects[i_found-1]
        o2 = self.objects[i_found]
        assert o1.frame_index < cur_frame_ind < o2.frame_index

        dindex = o2.frame_index - o1.frame_index
        d_cur_index1 = cur_frame_ind - o1.frame_index
        d_cur_index2 = o2.frame_index - cur_frame_ind

        bbox1 = o1.bbox
        bbox2 = o2.bbox

        res_bbox = [None, None, None, None]
        for k in range(4):
            # linear approximation for all bbox fields
            res_bbox[k] = (bbox1[k] * d_cur_index2 + bbox2[k] * d_cur_index1) / dindex
        res_bbox = Bbox(res_bbox[0], res_bbox[1], res_bbox[2], res_bbox[3])

        return res_bbox

class Tracker:
    def __init__(self, num_frames_to_remove_track, num_objects_to_make_track_valid, affinity_threshold):
        self.tracks = []
        self.track_archive = []
        self.num_frames_to_remove_track = num_frames_to_remove_track
        self.num_objects_to_make_track_valid = num_objects_to_make_track_valid
        self.affinity_threshold = affinity_threshold

    def add_objects(self, det_objs):
        log.debug("begin: handling {} objects".format(len(det_objs)))
        if len(det_objs) == 0:
            return

        frame_index = det_objs[0].frame_index
        assert all(o.frame_index == frame_index for o in det_objs), "All det_objs should have the same frame_index"

        affinity_matrix = self._build_affinity_matrix(det_objs)
        self._validate_affinity_matrix(affinity_matrix, len(self.tracks), len(det_objs))

        self._log_affinity_matrix(affinity_matrix)

        decision, best_affinity = self._solve_assignment_problem(affinity_matrix)
        self._log_decision(decision, best_affinity, det_objs, frame_index)

        self._apply_decision(decision, det_objs, frame_index)
        self._move_obsolete_tracks_to_archive(frame_index)
        log.debug("end: handling {} objects".format(len(det_objs)))

    @staticmethod
    def _validate_affinity_matrix(affinity_matrix, num_tracks, num_det_objs):
        assert isinstance(affinity_matrix, list)
        assert len(affinity_matrix) == num_tracks
        for affinity_row in affinity_matrix:
            assert isinstance(affinity_row, list)
            assert len(affinity_row) == num_det_objs
            assert all(isinstance(v, float) for v in affinity_row)
            assert all(v >= 0 for v in affinity_row)

    def _build_affinity_matrix(self, det_objs):
        affinity_matrix = []
        for t in self.tracks:
            affinity_row = []
            for o in det_objs:
                cur_affinity = self._calc_affinity(t, o)
                affinity_row.append(cur_affinity)

            affinity_matrix.append(affinity_row)

        return affinity_matrix

    def _calc_affinity(self, track, obj):
        affinity_appearance = self._calc_affinity_appearance(track, obj)
        affinity_position = self._calc_affinity_position(track, obj)
        affinity_shape = self._calc_affinity_shape(track, obj)
        return affinity_appearance * affinity_position * affinity_shape

    def _calc_affinity_appearance(self, track, obj):
        raise NotImplementedError("The function _calc_affinity_appearance  is not implemented -- implement it by yourself")

    def _calc_affinity_position(self, track, obj):
        raise NotImplementedError("The function _calc_affinity_position is not implemented -- implement it by yourself")

    def _calc_affinity_shape(self, track, obj):
        raise NotImplementedError("The function _calc_affinity_shape is not implemented -- implement it by yourself")

    @staticmethod
    def _log_affinity_matrix(affinity_matrix):
        with np.printoptions(precision=2, suppress=True, threshold=sys.maxsize, linewidth=sys.maxsize):
            log.debug("Affinity matrix =\n{}".format(np.array(affinity_matrix)))

    def _solve_assignment_problem(self, affinity_matrix):
        decision, best_affinity = solve_assignment_problem(affinity_matrix, self.affinity_threshold)
        return decision, best_affinity

    def _log_decision(self, decision, best_affinity, det_objs, frame_index):
        log.debug("Logging decision for frame index={}".format(frame_index))
        num_tracks = len(self.tracks)
        for track_index in range(num_tracks):
            assert track_index in decision
            obj_index = decision[track_index] # index of the object assigned to the track
            if obj_index is not None:
                assert 0 <= obj_index < len(det_objs)
                obj_bbox = det_objs[obj_index].bbox
            else:
                obj_bbox = None

            cur_best_affinity = best_affinity[track_index]
            if cur_best_affinity is not None:
                best_affinity_str = "{:.3f}".format(cur_best_affinity)
            else:
                best_affinity_str = str(cur_best_affinity)

            log.debug("track_index={}, track id={}, last_bbox={}, decision={}, best_affinity={} => {}".format(
                      track_index, self.tracks[track_index].get_id(),
                      self.tracks[track_index].last().bbox,
                      decision[track_index],
                      best_affinity_str,
                      obj_bbox))

    def _apply_decision(self, decision, det_objs, frame_index):
        set_updated_tracks_indexes = set()
        num_det_objs = len(det_objs)
        num_tracks = len(self.tracks)
        object_indexes_not_mapped_to_tracks = set(range(num_det_objs)) # all indexes from 0 to num_det_objs-1
        for track_index in range(num_tracks):
            assert track_index in decision

            obj_index = decision[track_index] # index of the object assigned to the track
            if obj_index is None:
                # no objects are mapped for this track
                continue

            assert 0 <= obj_index < num_det_objs
            if obj_index not in object_indexes_not_mapped_to_tracks:
                raise RuntimeError("ERROR: Algorithm assigned the object {} to several tracks".format(obj_index))

            object_indexes_not_mapped_to_tracks.remove(obj_index)

            o = det_objs[obj_index]
            self.tracks[track_index].add_object(o)

        # create new tracks for all the objects not mapped to tracks
        for obj_index in object_indexes_not_mapped_to_tracks:
            o = det_objs[obj_index]
            self._create_new_track(o)

    def _create_new_track(self, o):
        new_track = Track(o)
        self.tracks.append(new_track)
        log.debug("created new track: id={} object: frame_index={}, {}".format(
                  new_track.get_id(), o.frame_index, o.bbox))

    def _move_obsolete_tracks_to_archive(self, frame_index):
        new_tracks = []
        for t in self.tracks:
            last_frame_index = t.last().frame_index
            if frame_index - last_frame_index >= self.num_frames_to_remove_track:
                log.debug("Move the track id={} to archive: the current frame_index={}, "
                          "the last frame_index in track={}".format(
                              t.get_id(), frame_index, last_frame_index))
                self.track_archive.append(t)
            else:
                new_tracks.append(t)

        self.tracks = new_tracks

    def is_track_valid(self, track):
        assert isinstance(track, Track)
        return len(track.objects) > self.num_objects_to_make_track_valid

    def get_all_valid_tracks(self):
        res = []
        for t in self.track_archive:
            if self.is_track_valid(t):
                res.append(t)

        for t in self.tracks:
            if self.is_track_valid(t):
                res.append(t)

        return res

def convert_tracks_to_annotation_storage(tracks):
    ann_objects_by_frame_index = {}
    for cur_track in tqdm(tracks, desc="Converting"):
        track_id = cur_track.get_id()

        first_frame_index = cur_track.objects[0].frame_index
        last_frame_index = cur_track.objects[-1].frame_index

        for frame_index in range(first_frame_index, last_frame_index+1):
            bbox = cur_track.get_bbox_for_frame(frame_index)
            tl_x = math.floor(bbox.tl_x)
            tl_y = math.floor(bbox.tl_y)
            br_x = math.ceil(bbox.br_x)
            br_y = math.ceil(bbox.br_y)
            detect_obj = DetectedObject(frame_index=frame_index,
                                        bbox=Bbox(tl_x, tl_y, br_x, br_y),
                                        appearance_feature=[])
            ann_obj = AnnotationObject(detect_obj=detect_obj,
                                       track_id=track_id)
            if frame_index not in ann_objects_by_frame_index:
                ann_objects_by_frame_index[frame_index] = {}

            ann_objects_by_frame_index[frame_index][track_id] = ann_obj

    annotation_objects = []
    for frame_index in sorted(ann_objects_by_frame_index.keys()):
        cur_ann_objects = ann_objects_by_frame_index[frame_index]
        for track_id in sorted(cur_ann_objects.keys()):
            annotation_objects.append(cur_ann_objects[track_id])

    annotation_storage = AnnotationStorage.create_annotation_storage_from_list(annotation_objects)
    return annotation_storage
