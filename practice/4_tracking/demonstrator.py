from pathlib import Path
from common.colors import get_random_colors
from common.common_objects import get_bbox_center
import math
from tqdm import tqdm
import cv2

class Demonstrator:
    def __init__(self, images_folder_path, dst_folder, max_output_image_width, should_show):
        if images_folder_path is None or dst_folder is None:
            self.images_folder_path = images_folder_path
            self.dst_folder = None
            self.should_show = False
            return

        images_folder_path = Path(images_folder_path)
        images_folder_path = images_folder_path.resolve()
        self.images_folder_path = images_folder_path

        self.dst_folder = Path(dst_folder)
        self.dst_folder.mkdir(exist_ok=True)

        self.max_output_image_width = max_output_image_width
        self.should_show = should_show
        self.num_frame_indexes_in_track_centers = 100

    def make_demonstration(self, annotation_storage):
        if self.images_folder_path is None or self.dst_folder is None:
            return
        max_track_id = max(ann_obj.track_id for ann_obj in annotation_storage.all_objects)
        colors = get_random_colors(max_track_id+1)

        all_track_centers = {}
        first_frame_index = min(annotation_storage.get_list_of_frame_indexes())
        last_frame_index = max(annotation_storage.get_list_of_frame_indexes())

        frame_range = range(first_frame_index, last_frame_index+1)
        for frame_index in tqdm(frame_range, desc="Writing images"):
            cur_img_name = "{:06}.jpg".format(frame_index)

            src_img_path = self.images_folder_path / cur_img_name
            dst_img_path = self.dst_folder / cur_img_name

            img = cv2.imread(src_img_path.as_posix())

            ann_objects = annotation_storage.objects_by_frame.get(frame_index)
            if ann_objects is None:
                ann_objects = []

            for ann_obj in ann_objects:
                bbox = ann_obj.detect_obj.bbox
                if bbox is None:
                    continue

                track_id = ann_obj.track_id

                tl_x = math.floor(bbox.tl_x)
                tl_y = math.floor(bbox.tl_y)
                br_x = math.ceil(bbox.br_x)
                br_y = math.ceil(bbox.br_y)
                color = colors[track_id]

                cv2.rectangle(img, (tl_x, tl_y), (br_x, br_y), color, 2)
                cv2.putText(img,
                            str(track_id),
                            (tl_x, tl_y - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

                center = get_bbox_center(bbox)
                if track_id not in all_track_centers:
                    all_track_centers[track_id] = {}
                all_track_centers[track_id][frame_index] = center
                self._draw_track_centers(img, all_track_centers[track_id], color)

            self._remove_obsolete_track_centers(all_track_centers, frame_index)

            H, W = img.shape[:2]
            scale = self.max_output_image_width / W
            if scale < 1:
                H1 = math.floor(H * scale)
                W1 = self.max_output_image_width
                img = cv2.resize(img, (W1, H1))

            if self.should_show:
                cv2.imshow("tracking", img)
                cv2.waitKey(25)

            cv2.imwrite(dst_img_path.as_posix(), img)

    @staticmethod
    def _draw_track_centers(img, track_centers, color):
        frame_indexes = sorted(track_centers.keys())
        for i in range(1, len(frame_indexes)):
            frame_index_1 = frame_indexes[i-1]
            frame_index_2 = frame_indexes[i]
            center1 = track_centers[frame_index_1]
            center2 = track_centers[frame_index_2]
            center1 = tuple(int(v) for v in center1)
            center2 = tuple(int(v) for v in center2)
            cv2.line(img, center1, center2, color, 6)
            cv2.circle(img, center2, 4, color, 4)

    def _remove_obsolete_track_centers(self, all_track_centers, frame_index):
        frame_index_to_keep = frame_index - self.num_frame_indexes_in_track_centers
        for track_id in all_track_centers.keys():
            frame_indexes = sorted(all_track_centers[track_id].keys())
            frame_indexes_to_del = [i for i in frame_indexes if i < frame_index_to_keep]
            for frame_index in frame_indexes_to_del:
                del all_track_centers[track_id][frame_index]



