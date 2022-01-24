# draw vehicle flow direction warn line as VFDW_line
import cv2
import numpy as np
from collections import deque
import math
from ruamel import yaml

tpPointsChoose = deque(maxlen=2)
drawing = False
tempFlag = False


class DrawUtils():

    def __init__(self, YAML_NAME, is_conv, data) -> None:
        self.is_conv = is_conv
        self.YAML_NAME = YAML_NAME
        if not data:
            data = {
                'conversetrack':[],
                'sidetrack':[]
            }
        self.line_cord_dict = data

    def draw_ROI(self, event, x, y, flags, param):
        global point1, tpPointsChoose, pts, drawing, tempFlag
        if event == cv2.EVENT_LBUTTONDOWN:
            tempFlag = True
            drawing = False
            point1 = (x, y)
            tpPointsChoose.append((x, y))  # 用于画点
            # print(x,y)
        if event == cv2.EVENT_RBUTTONDOWN:
            tempFlag = True
            drawing = True
            (x_c, y_c), (x_o, y_o) = get_vertical_point_from_line_point(tpPointsChoose[0], tpPointsChoose[1], arr_len=50)
            if self.is_conv:
                cord_dict = {'arrowCord': [(x_c, y_c), (x_o, y_o)], 'lineCord': [tpPointsChoose[0], tpPointsChoose[1]]}
                self.line_cord_dict['conversetrack'].append(cord_dict)
            else:
                cord_dict = {
                    'lineCord': [tpPointsChoose[0], tpPointsChoose[1]],
                }
                self.line_cord_dict['sidetrack'].append(cord_dict)

            with open(self.YAML_NAME, "w", encoding="utf-8") as yaml_file:
                yaml.dump(self.line_cord_dict, yaml_file, Dumper=yaml.RoundTripDumper)
        if event == cv2.EVENT_MBUTTONDOWN:
            tempFlag = False
            drawing = True
            tpPointsChoose = []

    def run(self, VideoAddress):

        cv2.namedWindow('draw vehicle flow direction warn line.  ESC to ensure&Quit', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('draw vehicle flow direction warn line.  ESC to ensure&Quit', self.draw_ROI)
        cap = cv2.VideoCapture(VideoAddress)
        while (True):
            # capture frame-by-frame
            ret, frame = cap.read()
            # 画逆行线
            for conversetrack in self.line_cord_dict['conversetrack']:
                cv2.arrowedLine(frame, conversetrack['arrowCord'][0], conversetrack['arrowCord'][1], (0, 0, 255), thickness=2)
                cv2.line(frame, conversetrack['lineCord'][0], conversetrack['lineCord'][1], (255, 0, 0), 2)

            # 画变道线
            for sidetrack in self.line_cord_dict['sidetrack']:
                cv2.line(frame, sidetrack['lineCord'][0], sidetrack['lineCord'][1], (0, 255, 0), 2)

            # display the resulting frame
            if (tempFlag == True and drawing == False):  # 鼠标点击
                cv2.circle(frame, point1, 5, (0, 255, 0), 2)
                for i in range(len(tpPointsChoose) - 1):
                    cv2.line(frame, tpPointsChoose[i], tpPointsChoose[i + 1], (255, 0, 0), 2)
            if (tempFlag == True and drawing == True):  # 鼠标右击

                if self.is_conv:
                    (x_c, y_c), (x_o, y_o) = get_vertical_point_from_line_point(tpPointsChoose[0], tpPointsChoose[1], arr_len=50)
                    cv2.arrowedLine(frame, (x_c, y_c), (x_o, y_o), (0, 0, 255), thickness=2)
                    cv2.line(frame, tpPointsChoose[0], tpPointsChoose[1], (255, 0, 0), 2)
                else:
                    cv2.line(frame, tpPointsChoose[0], tpPointsChoose[1], (0, 255, 0), 2)

            if (tempFlag == False and drawing == True):  # 鼠标中键
                for i in range(len(tpPointsChoose) - 1):
                    cv2.line(frame, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)
            cv2.imshow('draw vehicle flow direction warn line.  ESC to ensure&Quit', frame)

            c = cv2.waitKey(5) & 0xff
            if c == 27:
                break
            elif cv2.waitKey(5) & 0xFF == ord('q'):
                break
        # when everything done , release the capture
        cap.release()
        cv2.destroyAllWindows()


def get_vertical_point_from_line_point(pointA, pointB, arr_len=20):
    '''
    根据两点得到相应line的垂线坐标
    Args:
        pointA: A点坐标
        pointB: B点坐标
        arr_len: 垂线箭头的长度
    return:
        (x_c, y_c), (x_o, y_o) 垂线箭头的起点和终点坐标(取整)
    '''
    x = pointB[0] - pointA[0]
    y = pointB[1] - pointA[1]
    x_c = (pointA[0] + pointB[0]) / 2
    y_c = (pointA[1] + pointB[1]) / 2
    theta = math.atan2(y, x)  # atan2返回(x,y)与x轴的弧度
    k = math.tan(theta)  # tan返回弧度的正切值(斜率)
    # 垂线的斜率
    if y == 0:  # 水平线的theta=0，他的垂线的斜率不存在
        dy = arr_len if x > 0 else -arr_len  # arr_len对应dy的长度
        x_o = x_c
        y_o = y_c + dy
    else:
        k2 = -1 / k
        atan = math.atan(k2)
        cos = math.cos(atan)
        dx_abs = arr_len * cos  # arr_len对应dx的长度的绝对值
        dx = dx_abs if x > 0 else -dx_abs
        x_o = x_c + dx
        y_o = k2 * (x_o - x_c) + y_c  # 垂线方程 点斜式 y_o-y_c = k2(x_o - x_c)
    return (round(x_c), round(y_c)), (round(x_o), round(y_o))


def read_yaml(YAML_NAME) -> dict:
    with open(YAML_NAME, 'r') as f:
        data = yaml.load(f.read(), Loader=yaml.Loader)
    return data


if __name__ == '__main__':
    is_conv = False  # 添加逆行线，False表示添加变道线
    VideoAddress = 'F:\DataSet\data_test_demo\欧亚三纬路01_Address.mp4'
    YAML_NAME = 'configs\line_cor.yaml'
    data = read_yaml(YAML_NAME)

    du = DrawUtils(YAML_NAME, is_conv, data)
    du.run(VideoAddress)
