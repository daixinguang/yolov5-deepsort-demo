import argparse
import cv2
import torch
import numpy as np
from pathlib import Path

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, increment_path, non_max_suppression,
                           print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync


class Yolov5():

    def __init__(self, args) -> None:
        self.device = args.device
        self.weights = args.weights
        self.data = args.data
        self.imgsz = args.imgsz
        self.conf_thres = args.conf_thres
        self.iou_thres = args.iou_thres
        self.classes = args.classes
        self.max_det = args.max_det
        self.view_img = args.view_img

        self.line_thickness = 3

        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=self.data)
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride)  # check image size
        self.names = self.model.names

        # Half
        self.half = False
        # half precision only supported by PyTorch on CUDA
        self.half &= (self.model.pt or self.model.jit or self.model.engine) and self.device.type != 'cpu'
        if self.model.pt or self.model.jit:
            self.model.model.half() if self.half else self.model.model.float()

        # Run inference
        self.model.warmup(imgsz=(1, 3, *self.imgsz), half=self.half)  # warmup

    def get_dataset(self):
        return self.dataset

    @torch.no_grad()
    def run(self, path, im, im0s, vid_cap):
        '''
        使用LoadImages的返回值作为函数的输入
        Example:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            for path, im, im0s, vid_cap, s in dataset:
                bbox_xywhs, confs, clas, xyxys = det.run(path, im, im0s, vid_cap)
        '''
        dt = [0.0, 0.0, 0.0]
        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = self.model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, agnostic=False,
                                   max_det=self.max_det)  # agnostic: agnostic_nms
        dt[2] += time_sync() - t3

        # Process predictions
        bbox_xywhs = []
        confs = []
        clas = []
        xyxys = []
        for i, det in enumerate(pred):  # per image
            im0 = im0s.copy()
            # p = Path(p)  # to Path
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                    xyxy_list = torch.tensor(xyxy).view(1, 4).squeeze().tolist()
                    bbox_xywhs.append(xywh)
                    confs.append(conf.item())
                    clas.append(self.names[int(cls)])
                    xyxys.append(xyxy_list)
                    # print('xywh={}, conf={}, cls={}, xyxy={}'.format(xywh, conf, cls, xyxy_list))
                    c = int(cls)  # integer class
                    label = (f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Print time (inference-only)
            LOGGER.info(f'Detect Done. ({t3 - t2:.3f}s)')
            # Stream results
            im0 = annotator.result()
            if self.view_img:
                cv2.namedWindow('VIEW', cv2.WINDOW_NORMAL)
                cv2.resizeWindow("VIEW", int(1920 / 2), int(1080 / 2))
                cv2.imshow('VIEW', im0)
                # cv2.waitKey(1)  # 1 millisecond
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
        return np.array(bbox_xywhs), confs, clas, xyxys


def parse_args():
    parser = argparse.ArgumentParser()
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
    ''' 语法糖
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1
    等价于
    if (len(args.imgsz) == 1):
        args.imgsz *= 2
    else:
        continue
    '''
    return args


if __name__ == '__main__':
    video_path = 'F:\DataSet\data_test_demo\欧亚三纬路002.jpg'
    args = parse_args()
    det = Yolov5(args)
    # Load dataset
    dataset = LoadImages(video_path, img_size=det.imgsz, stride=det.model.stride, auto=det.model.pt)
    for path, im, im0s, vid_cap, s in dataset:
        bbox_xywhs, confs, clas, xyxys = det.run(path, im, im0s, vid_cap)
        print('bbox_xywhs={}, confs={}, clas={}, xyxys={}'.format(bbox_xywhs, confs, clas, xyxys))
'''

********************测试用********************
detect_test_py
```shell
执行detect_test.py
$ python detect_test.py --view-img
```

关于video_path:
- 欧亚三纬路002.jpg
    图片按q关闭
- F:\DataSet\data_test_demo\欧亚三纬路01_Address.mp4
    视频一直按q，直到所有帧都关闭程序结束，或者直接按两次(Ctrl + C) 终止程序

********************测试用********************
'''
