from pathlib import Path
import logging as log
import numpy as np
import sys
from common.common_objects import DetectedObject, validate_detected_object, Bbox, calc_IoU

def _find_argmax_from_list(dictionary, list_keys):
    if len(list_keys) == 0:
        return None
    max_el = max( (dictionary[k], k) for k in list_keys )
    best_key = max_el[1]
    return best_key

class Evaluator:
    def __init__(self, annotation_storage, max_frame_index):
        self.annotation_storage = annotation_storage
        self.max_frame_index = max_frame_index

        self.gt_by_track_id = Evaluator._convert_annot_to_dict_by_track_id(annotation_storage,
                                                                           self.max_frame_index)

        self.iou_threshold = 0.5
        self.min_relative_num_frames_to_count_gt_track_hit = 0.75

    @staticmethod
    def _convert_annot_to_dict_by_track_id(annotation_storage, max_frame_index):
        objects_by_track_id = {}
        for ann_obj in annotation_storage.all_objects:
            frame_index = ann_obj.detect_obj.frame_index
            if frame_index > max_frame_index:
                continue

            track_id = ann_obj.track_id

            if track_id not in objects_by_track_id:
                objects_by_track_id[track_id] = []

            objects_by_track_id[track_id].append(ann_obj)
        return objects_by_track_id

    @staticmethod
    def _calc_affinity_res_and_gt(res_ann_objects, gt_ann_objects, iou_threshold):
        # note that all our tracks are continuous
        min_res_frame_index = res_ann_objects[0].detect_obj.frame_index
        max_res_frame_index = res_ann_objects[-1].detect_obj.frame_index
        num_res_frames = max_res_frame_index - min_res_frame_index + 1
        assert len(res_ann_objects) == num_res_frames

        min_gt_frame_index = gt_ann_objects[0].detect_obj.frame_index
        max_gt_frame_index = gt_ann_objects[-1].detect_obj.frame_index
        num_gt_frames = max_gt_frame_index - min_gt_frame_index + 1
        assert len(gt_ann_objects) == num_gt_frames

        if min_gt_frame_index > min_res_frame_index:
            # result track is started earlier than gt track
            return 0, 0

        if max_gt_frame_index < max_res_frame_index:
            # result track is finished after than gt track
            return 0, 0

        first_ind = None
        for i, ann_obj in enumerate(gt_ann_objects):
            if ann_obj.detect_obj.frame_index == min_res_frame_index:
                first_ind = i
                break
        if first_ind is None:
            return 0, 0

        num_good = 0
        for i in range(num_res_frames):
            # this is possible since we have continuous tracks only
            assert res_ann_objects[i].detect_obj.frame_index == gt_ann_objects[first_ind + i].detect_obj.frame_index

            iou = calc_IoU(res_ann_objects[i].detect_obj.bbox,
                           gt_ann_objects[first_ind + i].detect_obj.bbox)

            if iou >= iou_threshold:
                num_good += 1

        cur_affinity_res_to_gt = num_good / num_res_frames
        cur_affinity_gt_to_res = num_good / num_gt_frames
        return cur_affinity_res_to_gt, cur_affinity_gt_to_res

    @staticmethod
    def _find_best_correspondence(affinity_res_to_gt, affinity_gt_to_res):
        res_track_ids = sorted(affinity_res_to_gt.keys())
        gt_track_ids = sorted(affinity_gt_to_res.keys())

        # find for each res_track_id the best gt_track_id
        decision_res_to_gt = {}
        for res_track_id in res_track_ids:
            best_gt_track_id = _find_argmax_from_list(affinity_res_to_gt[res_track_id],
                                                      gt_track_ids)

            decision_res_to_gt[res_track_id] = best_gt_track_id
        Evaluator._log_decision("In _find_argmax_from_list decision_res_to_gt=", decision_res_to_gt)

        # build map {gt_track_id => list all res_track_id-s pointing to this gt track}
        decision_gt_to_many_res = {}
        for gt_track_id in gt_track_ids:
            decision_gt_to_many_res[gt_track_id] = []

        for res_track_id in res_track_ids:
            gt_track_id = decision_res_to_gt[res_track_id]
            if gt_track_id is None:
                continue
            decision_gt_to_many_res[gt_track_id].append(res_track_id)

        Evaluator._log_decision("In _find_argmax_from_list decision_gt_to_many_res=", decision_gt_to_many_res)

        # find for each gt_track_id the best res_track_id
        # from the list decision_gt_to_many_res[gt_track_id]
        decision_gt_to_res = {}
        for gt_track_id in gt_track_ids:
            best_res_id = _find_argmax_from_list(affinity_gt_to_res[gt_track_id],
                                                 decision_gt_to_many_res[gt_track_id])
            decision_gt_to_res[gt_track_id] = best_res_id
        Evaluator._log_decision("In _find_argmax_from_list decision_gt_to_res=", decision_gt_to_res)
        return decision_gt_to_res

    def evaluate(self, result_annotation_storage, dst_file_path=None):
        result_by_track_id = Evaluator._convert_annot_to_dict_by_track_id(result_annotation_storage,
                                                                          self.max_frame_index)
        gt_track_ids = sorted(self.gt_by_track_id.keys())
        res_track_ids = sorted(result_by_track_id.keys())

        affinity_res_to_gt = {}
        affinity_gt_to_res = {}
        for res_track_id in res_track_ids:
            affinity_res_to_gt[res_track_id] = {}
        for gt_track_id in gt_track_ids:
            affinity_gt_to_res[gt_track_id] = {}

        for res_track_id in res_track_ids:
            for gt_track_id in gt_track_ids:
                cur_affinity_res_to_gt, cur_affinity_gt_to_res = \
                        self._calc_affinity_res_and_gt(result_by_track_id[res_track_id],
                                                       self.gt_by_track_id[gt_track_id],
                                                       self.iou_threshold)
                affinity_res_to_gt[res_track_id][gt_track_id] = cur_affinity_res_to_gt
                affinity_gt_to_res[gt_track_id][res_track_id] = cur_affinity_gt_to_res

        self._log_affinity_matrix("affinity_res_to_gt", affinity_res_to_gt)
        self._log_affinity_matrix("affinity_gt_to_res", affinity_gt_to_res)

        decision_gt_to_res = self._find_best_correspondence(affinity_res_to_gt, affinity_gt_to_res)

        num_hits = 0
        num_misses = 0
        for gt_track_id in gt_track_ids:
            res_track_id = decision_gt_to_res[gt_track_id]
            log.debug("For gt_track_id={} res_track_id={}".format(gt_track_id, res_track_id))
            if res_track_id is None:
                log.debug("  it is miss")
                num_misses += 1
                continue

            relative_num_matched_frames_in_gt = affinity_gt_to_res[gt_track_id][res_track_id]
            is_hit = (relative_num_matched_frames_in_gt >= self.min_relative_num_frames_to_count_gt_track_hit)

            log.debug("  relative_num_matched_frames_in_gt={}".format(relative_num_matched_frames_in_gt))
            log.debug("  is_hit={}".format(is_hit))

            if is_hit:
                num_hits += 1
            else:
                num_misses += 1

        recall = num_hits / (num_hits + num_misses)

        if dst_file_path is not None:
            with Path(dst_file_path).open("w") as f_dst:
                f_dst.write("num_hits: {}\n".format(num_hits))
                f_dst.write("num_misses: {}\n".format(num_misses))
                f_dst.write("recall: {:.4f}\n".format(recall))
        return num_hits, num_misses, recall

    @staticmethod
    def _log_affinity_matrix(name, affinity_matrix):
        if len(affinity_matrix) == 0:
            log.debug("Affinity matrix '{}' is empty".format(name))
        rows = sorted(affinity_matrix.keys())
        cols = sorted(affinity_matrix[rows[0]].keys())
        if len(cols) == 0:
            log.debug("Affinity matrix '{}' has empty rows".format(name))

        affinity_matrix = [ [affinity_matrix[r][c] for c in cols] for r in rows ]
        with np.printoptions(precision=2, suppress=True, threshold=sys.maxsize, linewidth=sys.maxsize):
            log.debug("Affinity matrix '{}' rows = \n{}".format(name, np.array(rows)))
            log.debug("Affinity matrix '{}' cols = \n{}".format(name, np.array(cols)))
            log.debug("Affinity matrix '{}' =\n{}".format(name, np.array(affinity_matrix)))

    @staticmethod
    def _log_decision(header, decision):
        log_str = "\n".join("{:04} => {}".format(k, decision[k]) for k in sorted(decision.keys()))
        log.debug("{}\n{}".format(header, log_str))
