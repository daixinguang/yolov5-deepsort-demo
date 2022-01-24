import os
import cv2
import time
import math
from ruamel import yaml
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

    def __init__(self, args, cfg, violation_data) -> None:

        # lineCord conversetrack：逆行线，sidetrack：变道线
        self.line_cord_dict = violation_data
        # self.line_cord_dict = {
        #     'conversetrack': [{
        #         'arrowCord': [(439, 721), (445, 771)],
        #         'lineCord': [(232, 748), (646, 694)]
        #     }, {
        #         'arrowCord': [(360, 460), (351, 410)],
        #         'lineCord': [(447, 445), (272, 474)]
        #     }],
        #     'sidetrack': [{
        #         'lineCord': [(242, 357), (648, 694)]
        #     }]
        # }

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

                # 逆行判断
                self.line_cord_dict
                if len(trail_deque) > 1:
                    is_viola = False
                    is_viola = violation_detect(trail_deque, self.line_cord_dict['conversetrack'], self.line_cord_dict['sidetrack'])
                    if is_viola:
                        print('''
//----------------------------//

            WARNING!!!

//----------------------------//
                            ''')
                # 画逆行线
                for conversetrack in self.line_cord_dict['conversetrack']:
                    cv2.arrowedLine(ori_img, conversetrack['arrowCord'][0], conversetrack['arrowCord'][1], (0, 0, 255), thickness=2)
                    cv2.line(ori_img, conversetrack['lineCord'][0], conversetrack['lineCord'][1], (255, 0, 0), 2)
                
                # 画变道线
                for sidetrack in self.line_cord_dict['sidetrack']:
                    cv2.line(ori_img, sidetrack['lineCord'][0], sidetrack['lineCord'][1], (0, 255, 0), 2)
            t2 = time_sync()

            if self.args.display:
                cv2.namedWindow("test", cv2.WINDOW_NORMAL)
                cv2.imshow("test", ori_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # logging
            # self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                            #  .format(t2 - t1, 1 / (t2 - t1), bbox_xywhs.shape[0], len(outputs)))


def violation_detect(trail_deque, conversetrack_list, sidetrack_list) -> bool:
    '''
    逆行return Ture
    '''
    is_cross = False
    trail_dict = dict()
    if len(trail_deque) > 1:
        for i in range(len(trail_deque) - 1):
            new = trail_deque[i]
            old = trail_deque[i + 1]
            for key, value in new.items():
                if key in old.keys():
                    old_point = old[key]
                    new_point = value
                    # 获取车辆行驶方向的数据准备
                    if key not in trail_dict.keys():
                        trail_dict[key] = deque(maxlen=20)  # 取连续前五帧的均值作为车辆行驶方向
                    else:
                        # if i % 10 == 0:  # 50个轨迹点取5个点
                        trail_dict[key].append(new_point)
        # direction
        # print(trail_dict)
        for key, value in trail_dict.items():
            if len(trail_dict[key]) > 1:
                # value是50个轨迹点遍历后的(x,y)
                vehicle_line = [value[0], value[1]]
                for sidetrack in sidetrack_list:
                    # 是否有车辆违法变道
                    is_side = intersection(sidetrack['lineCord'], vehicle_line)
                    if is_side:
                        return True
                for conversetrack in conversetrack_list:
                    # 是否有车辆穿过逆行线
                    is_cross = intersection(conversetrack['lineCord'], vehicle_line)
                    if is_cross:
                        count = 0
                        angle = 0
                        vehicle_angle = 0
                        for i in range(len(value) - 1):
                            count += 1
                            drn_new = value[i]
                            drn_old = value[i + 1]
                            x = drn_new[0] - drn_old[0]
                            y = drn_new[1] - drn_old[1]
                            # out->new的角度，使用math.atan2(y, x) x和y的增量得到角度
                            angle += math.atan2(y, x)
                            # print('{} : angle=={}'.format(key, angle/math.pi))
                        if count != 0:
                            angle = angle / count
                            vehicle_angle = math.degrees(angle)
                            # print('{} : vehicle_angle=={:.2f}'.format(key, angle))
                        # print('{} : avg_vehicle_angle=={}'.format(key, vehicle_angle))

                        x2 = conversetrack['arrowCord'][1][0] - conversetrack['arrowCord'][0][0]
                        y2 = conversetrack['arrowCord'][1][1] - conversetrack['arrowCord'][0][1]
                        arrow_angle = math.degrees(math.atan2(y2, x2))
                        # print('arrow_angle: {}'.format(arrow_angle))
                        if abs(vehicle_angle - arrow_angle) > 60:
                            return True
    return False


def intersection(line1, line2) -> bool:
    '''
    line1 = [(),()]
    line2 = [(),()]
    两线段相交返回True， 不相交返回False
    https://blog.csdn.net/qq826309057/article/details/70942061
    两次判断，先快速排斥实验，再跨立实验
    '''
    A, B = line1
    C, D = line2
    '''
    1.快速排斥实验
    我们首先判断两条线段在 x 以及 y 坐标的投影是否有重合。
    也就是判断下一个线段中 x 较大的端点是否小于另一个线段中 x 较小的段点，若是，则说明两个线段必然没有交点，同理判断下 y。
    代码就实现就是有一个不满足就False
    '''
    if max(A[0], B[0]) < min(C[0], D[0]) or \
        min(A[0], B[0]) > max(C[0], D[0]) or \
            max(A[1], B[1]) < min(C[1], D[1]) or \
                min(A[1], B[1]) > max(C[1], D[1]):
        return False
    '''
    2.跨立实验
        矢量叉积
        计算矢量叉积是与直线和线段相关算法的核心部分。
        设矢量 P = (x1, y1)，Q = ( x2, y2 )，则矢量叉积定义为：P × Q = x1*y2 - x2*y1，其结果是一个矢量，与为 P Q 向量所在平面的法向量。显然有性质 P × Q = - ( Q × P ) 和 P × ( - Q ) = - ( P × Q )。

        叉积的一个非常重要性质是可以通过它的符号判断两矢量相互之间的顺逆时针关系：
        若 P × Q > 0 , 则 P 在 Q 的顺时针方向。
        若 P × Q < 0 , 则 P 在 Q 的逆时针方向。
        若 P × Q = 0 , 则 P 与 Q 共线，但可能同向也可能反向。

        如果两线段相交那么就意味着它们互相跨立，即如上图点 A 和 B 分别在线段 CD 两侧，点 C 和 D 分别在线 AB 两侧。
        判断 A 点与 B 点是否在线段 DC 的两侧，即向量 A-D 与向量 B-D 分别在向量 C-D 的两端，也就是其叉积是异号的，
        即 (A−D)×(C−D)∗(B−D)×(C−D)<0(A−D)×(C−D)∗(B−D)×(C−D)<0。
        同时也要证明 C 点与 D 点在线段 AB 的两端，两个同时满足，则表示线段相交。
        临界情况下(A−D)×(C−D)∗(B−D)×(C−D)=0,也相交。所以就是小于等于零的情况下相交

        代码就实现就是有一个不满足就False
    '''
    AB = (B[0] - A[0], B[1] - A[1])
    CD = (D[0] - C[0], D[1] - C[1])
    AD = (D[0] - A[0], D[1] - A[1])
    BD = (D[0] - B[0], D[1] - B[1])
    DB = (B[0] - D[0], B[1] - D[1])
    CB = (B[0] - C[0], B[1] - C[1])
    '''
    AD x CD = AD[0]*CD[1] - CD[0]*AD[1]
    DB x AB = DB[0]*AB[1] - AB[0]*DB[1]
    AD x CD * BD x CD > 0 or DB x AB * CB x AB > 0
    '''
    if (AD[0]*CD[1] - CD[0]*AD[1]) * (BD[0]*CD[1] - CD[0]*BD[1]) > 0 or \
        (DB[0]*AB[1] - AB[0]*DB[1]) * (CB[0]*AB[1] - AB[0]*CB[1]) > 0:
        return False
    return True

def read_yaml(YAML_NAME) -> dict:
    with open(YAML_NAME, 'r') as f:
        data = yaml.load(f.read(), Loader=yaml.Loader)
    return data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", default='F:\DataSet\data_test_demo\欧亚三纬路01_Address.mp4', type=str)
    parser.add_argument("--violation-data", default='configs\line_cor.yaml')
    # deep_sort
    parser.add_argument("--sort", default=True, help='True: sort model, False: reid model')
    parser.add_argument("--config-deepsort", type=str, default="deep_sort\configs\deep_sort.yaml")
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
    violation_data = read_yaml(args.violation_data)
    yd = YoloDeepsort(args, cfg, violation_data)
    yd.deep_sort()
