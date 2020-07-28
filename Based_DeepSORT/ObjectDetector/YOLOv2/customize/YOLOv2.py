import numpy as np
from darkflow.net.build import TFNet


class YOLOv2:
    def __init__(self, model, weight, threshold, gpu, labels):
        self.yolo = TFNet({"model": model, "load": weight, "threshold": threshold, "gpu": gpu, "labels": labels})

    def __del__(self):
        self.yolo.sess.close()

    def Detect(self, frame):
        detects = []
        for detect_yolo in self.yolo.return_predict(frame):
            if ("person" != detect_yolo['label']):
                continue

            detects.append([detect_yolo['topleft']['x'],
                            detect_yolo['topleft']['y'],
                            detect_yolo['bottomright']['x'],
                            detect_yolo['bottomright']['y'],
                            detect_yolo['confidence']])
        return np.array(detects)
