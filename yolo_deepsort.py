import os
import cv2
import time
import torch
import argparse
import numpy as np
# count
from collections import Counter
from collections import deque

from deep_sort.utils.parser import get_config
from yolov5 import build_detector
from yolov5.utils.datasets import LoadImages
from yolov5.utils.torch_utils import time_sync
from deep_sort import build_tracker
from deep_sort.utils.draw import draw_boxes
from deep_sort.utils.io import write_results
from deep_sort.utils.log import get_logger


class YoloDeepsort():

    def __init__(self, args, cfg) -> None:
        self.args = args
        self.logger = get_logger("root")
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            raise UserWarning("Running in cpu mode!")

        self.vdo = cv2.VideoCapture()

        self.yolo_detect = build_detector(args)
        self.dataset = LoadImages(args.video_path,
                                  img_size=self.yolo_detect.imgsz,
                                  stride=self.yolo_detect.model.stride,
                                  auto=self.yolo_detect.model.pt)
        self.deepsort_track = build_tracker(cfg, use_cuda=use_cuda)

    @torch.no_grad()
    def deep_sort(self):
        idx_frame = 0
        trail_deque = deque(maxlen=50)  # 车辆轨迹;deque始终保持maxlen最大长度的元素，如果超过了就会自动把以前的元素弹出(删除)
        # dict_box = dict()

        # path, im, im0s, vid_cap, s in in self.dataset
        for video_path, img, ori_img, vid_cap, s in self.dataset:
            idx_frame += 1
            # print('aaaaaaaa', video_path, img.shape, im0s.shape, vid_cap)
            t1 = time_sync()

            # yolo detection
            bbox_xywhs, confs, clas, xyxys = self.yolo_detect.run(video_path, img, ori_img, vid_cap)

            # do tracking
            outputs = self.deepsort_track.update(bbox_xywhs, confs, clas, ori_img)
            # outputs = [x1, y1, x2, y2, class, track_id]

            # draw boxes for visualization
            if len(outputs) > 0:
                # bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]  # 提取前四列  坐标
                cls = outputs[:, -2]  # 提取倒数第二列 class类别
                identities = outputs[:, -1]  # 提取最后一列 ID

                trail_dict = dict()
                for i, box in enumerate(bbox_xyxy):
                    x_c = (int(box[0]) + int(box[2])) // 2
                    # y_c = (int(box[1]) + int(box[3])) // 2
                    y2 = int(box[3])
                    center = (x_c, y2)
                    trail_dict[identities[i]] = center
                trail_deque.appendleft(trail_dict)  # 左进右出
                
                # 画框 + 轨迹
                ori_img = draw_boxes(ori_img, bbox_xyxy, cls, trail_deque, identities)
            t2 = time_sync()

            if self.args.display:
                cv2.namedWindow("test", cv2.WINDOW_NORMAL)
                cv2.imshow("test", ori_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(t2 - t1, 1 / (t2 - t1), bbox_xywhs.shape[0], len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default='F:\DataSet\data_test_demo\欧亚三纬路01_Address.mp4', type=str)
    # deep_sort
    parser.add_argument("--sort", default=True, help='True: sort model, False: reid model')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort\configs\deep_sort.yaml")
    parser.add_argument("--display", default=True, help='show resule')
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    # yolov5
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5\weights\yolov5ltruck2000.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='yolov5\configs\mydata.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--classes', default=[0], type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--view-img', action='store_true', help='show results')
    args = parser.parse_args()
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand
    return args


if __name__ == '__main__':
    args = parse_args()
    args.USE_YOLOV5 = True

    cfg = get_config()
    cfg.USE_FASTREID = False
    cfg.merge_from_file(args.config_deepsort)
    yd = YoloDeepsort(args, cfg)
    yd.deep_sort()
