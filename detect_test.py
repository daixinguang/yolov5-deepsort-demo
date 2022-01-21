
from yolov5.detect_yolov5 import Yolov5, parse_args
from yolov5.utils.datasets import LoadImages

if __name__ == '__main__':
    video_path = 'F:\DataSet\data_test_demo\欧亚三纬路01_Address.mp4'
    args = parse_args()
    det = Yolov5(args)
    # Load dataset
    dataset = LoadImages(video_path, img_size=det.imgsz, stride=det.model.stride, auto=det.model.pt)
    for path, im, im0s, vid_cap, s in dataset:
        bbox_xywhs, confs, clas, xyxys = det.run(path, im, im0s, vid_cap)
        print('bbox_xywhs={}, confs={}, clas={}, xyxys={}'.format(bbox_xywhs, confs, clas, xyxys))