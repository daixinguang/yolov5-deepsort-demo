import numpy as np
import cv2
import math
from collections import deque

palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label**2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, clss, trail_deque, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        cls = str(clss[i]) if clss is not None else ''
        color = compute_color_for_labels(id)
        label = '{}-{:d}'.format(cls, id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        # trail
        # trail_dict = dict()
        if len(trail_deque) > 1:
            for i in range(len(trail_deque) - 1):
                new = trail_deque[i]
                old = trail_deque[i + 1]
                for key, value in new.items():
                    if key in old.keys():
                        old_point = old[key]
                        new_point = value
                        cv2.line(img, new_point, old_point, color=(144, 238, 144), thickness=4, lineType=8)
        '''
                        # 获取车辆行驶方向的数据准备
                        if key not in trail_dict.keys():
                            trail_dict[key] = deque(maxlen=5)  # 取连续前五帧的均值作为车辆行驶方向
                        else:
                            if i % 10 == 0:  # 50个轨迹点取5个点
                                trail_dict[key].append(new_point)

        # direction
        # print(trail_dict)
        for key, value in trail_dict.items():
            count = 0
            angle = 0
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
                print('{} : angle=={:.2f}°'.format(key, math.degrees(angle)))
            # print('{} : avg_angle=={}'.format(key, angle))
        '''
    return img


if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
