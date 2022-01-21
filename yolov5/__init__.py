from yolov5.detect_yolov5 import Yolov5

__all__ = ['build_detector']

def build_detector(args):
    if args.USE_YOLOV5:
        return Yolov5(args)