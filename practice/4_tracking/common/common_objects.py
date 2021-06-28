from collections import namedtuple
import math

Bbox = namedtuple("Bbox", ["tl_x", "tl_y", "br_x", "br_y"])
DetectedObject = namedtuple("DetectedObject", ["frame_index", "bbox", "appearance_feature"])

def validate_bbox(bb):
    assert isinstance(bb, Bbox)
    for k in range(4):
        assert isinstance(bb[k], (int, float))

def validate_detected_object(o):
    assert isinstance(o, DetectedObject)
    assert isinstance(o.frame_index, int)

    # appearance_feature should be a list of floats
    assert isinstance(o.appearance_feature, list)
    assert all(v >= 0 for v in o.appearance_feature)

    # bbox should be a valid bbox
    validate_bbox(o.bbox)

def get_bbox_size(bb):
    validate_bbox(bb)
    width = bb.br_x - bb.tl_x
    height = bb.br_y - bb.tl_y
    return (width, height)

def get_bbox_center(bb):
    validate_bbox(bb)
    center_x = (bb.br_x + bb.tl_x) * 0.5
    center_y = (bb.br_y + bb.tl_y) * 0.5
    return (center_x, center_y)

def calc_bbox_area(bb):
    width, height = get_bbox_size(bb)
    if width < 0:
        width = 0
    if height < 0:
        height = 0
    return width * height

def get_dist(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x1-x2
    dy = y1-y2
    return math.sqrt(dx*dx + dy*dy)

def intersect_bboxes(bb1, bb2):
    tl_x_1, tl_y_1, br_x_1, br_y_1 = bb1
    tl_x_2, tl_y_2, br_x_2, br_y_2 = bb2
    tl_x = max(tl_x_1, tl_x_2)
    tl_y = max(tl_y_1, tl_y_2)
    br_x = min(br_x_1, br_x_2)
    br_y = min(br_y_1, br_y_2)
    return Bbox(tl_x, tl_y, br_x, br_y)

def calc_IoU(bb1, bb2):
    intersect = intersect_bboxes(bb1, bb2)
    area_intersect = calc_bbox_area(intersect)
    area1 = calc_bbox_area(bb1)
    area2 = calc_bbox_area(bb2)
    return area_intersect / (area1 + area2 - area_intersect)
