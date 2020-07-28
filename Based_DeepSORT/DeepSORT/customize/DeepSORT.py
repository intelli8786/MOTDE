import numpy as np
import cv2

from DeepSORT.tools import generate_detections
from DeepSORT.deep_sort.tracker import Tracker
from DeepSORT.deep_sort.detection import Detection
from DeepSORT.deep_sort import nn_matching
from DeepSORT.application_util import preprocessing

class DeepSORT(Tracker):
    def __init__(self, modelPath="", configs={}, models={}):
        self.max_cosine_distance = 0.3
        self.nn_budget = None
        self.nms_max_overlap = 1.0
        self.encoder = generate_detections.create_box_encoder(modelPath, batch_size=1)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        super().__init__(metric)

        self.configs = configs
        self.models = models

    def clear(self):
        self.tracks = []
        self._next_id = 1

    def update(self, frame):
        # Detect
        detects_candidate = self.models["personDetect"].Detect(frame)

        if len(detects_candidate) != 0:
            detects = detects_candidate[detects_candidate[:, 4] > self.configs["detect_threshold"]]

            # Re Detect
            if self.configs.get("redetect") is not None and self.configs["redetect"]:
                detects_refind = detects_candidate[(detects_candidate[:, 4] <= self.configs["detect_threshold"]) &
                                        (detects_candidate[:, 4] > self.configs["redetect_target_min"]) &
                                        ((detects_candidate[:, 2] - detects_candidate[:, 0]) * (detects_candidate[:, 3] - detects_candidate[:, 1]) < 480000)]
                for detect in detects_refind:
                    # Super Resolution & Deblurring
                    frame_redetect = frame[int(detect[1]):int(detect[3]), int(detect[0]):int(detect[2])]
                    frame_redetect = self.models["superResolution"].SuperResolution(frame_redetect)

                    # Re Detect
                    redetects_candidate = self.models["personDetect"].Detect(frame_redetect)
                    if len(redetects_candidate) == 0:
                        continue
                    redetects = redetects_candidate[redetects_candidate[:, 4] > self.configs["redetect_threshold"]]
                    if len(redetects) == 0:
                        continue
                    redetects[:, 0:4] = redetects[:, 0:4] / 4
                    redetects[:, 0:4] = redetects[:, 0:4] + [detect[0], detect[1], detect[0], detect[1]]
                    detects = np.vstack((detects, redetects))

            # Blur Classify & Deblurring
            if self.configs.get("recovery_body") is not None and self.configs["recovery_body"]:
                for detect in detects:
                    crop_x1,crop_y1,crop_x2,crop_y2 = [int(detect[0]),int(detect[1]), int(detect[2]),int(detect[3])]
                    print((crop_x2-crop_x1),(crop_y2 - crop_y1))
                    if (crop_x2-crop_x1) * (crop_y2 - crop_y1) > 480000:
                        continue

                    frame_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if cv2.Laplacian(frame_crop, cv2.CV_64F).var() < self.configs["recovery_body_threshold"]:
                        if self.configs["recovery_body_method"] == "sr":
                            frame_crop = self.models["superResolution"].SuperResolution(frame_crop)
                            frame[crop_y1:crop_y2, crop_x1:crop_x2] = cv2.resize(frame_crop, (crop_x2-crop_x1, crop_y2-crop_y1), interpolation=cv2.INTER_AREA)
                            print("SR Recovery")
                        elif self.configs["recovery_body_method"] == "deblur":
                            frame_crop = self.models["deblur"].Deblur(frame_crop)
                            frame[crop_y1:crop_y2, crop_x1:crop_x2] = frame_crop #cv2.resize(frame_crop, (crop_x2-crop_x1, crop_y2-crop_y1), interpolation=cv2.INTER_AREA)
                            print("Deblur Recovery")
                        else:
                            print("Recovery Parameter Error!")

        else:
            detects = np.array([])


        if len(detects) != 0:
            # xyxy -> xywh
            detects = detects.copy()
            detects[:, 2:4] = detects[:, 2:4] - detects[:, 0:2]

            # feature extract
            features = self.encoder(frame, detects[:, :4])

            params = [Detection(bbox, 1.0, feature) for bbox, feature in zip(detects[:, :4], features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in params])
            scores = np.array([d.confidence for d in params])
            indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
            params = [params[i] for i in indices]
        else:
            params = []


        detects_face = []
        if self.configs.get("faceRecognition") is not None and self.configs["faceRecognition"]:
            # 위 deblurring때문에 얼굴인식 성능이 떨어진다면 얼굴인식 frame은 원본을 사용하도록한다.
            detects_face = self.models["faceDetect"].Detect(frame)

            for detection in params:
                person_x1, person_y1, person_x2, person_y2 = detection.to_tlbr()
                faces = []
                for detect_face in detects_face:
                    face_x1, face_y1, face_x2, face_y2 = detect_face[:4]

                    if person_x1 < face_x1 and person_y1 < face_y1 and person_x2 > face_x2 and person_y2 > face_x2:
                        faces.append(detect_face)
                if len(faces) == 1:
                    crop_x1,crop_y1,crop_x2,crop_y2 = [int(faces[0][0]), int(faces[0][1]), int(faces[0][2]), int(faces[0][3])]
                    frame_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                    if self.configs.get("recovery_face") is not None and self.configs["recovery_face"]:
                        if max([face_x2-face_x1, face_y2-face_y1]) < self.configs["recovery_face_threshold"]:
                            frame_crop = self.models["superResolution"].SuperResolution(frame_crop)

                    detection.faceFeature = self.models["faceEmbed"].faceEmbedding(frame_crop)



        super().predict()
        super().update(params)

        regions = []
        ids = []
        for track in self.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            regions.append(track.to_tlbr())
            ids.append(track.track_id)

        # xywh -> xyxy
        if len(detects) != 0:
            detects[:,2:4] = detects[:, 2:4] + detects[:, 0:2]
        
        return regions, ids, detects, detects_face


def nms(boxes, probs, threshold):
  """Non-Maximum supression.
  Args:
    boxes: array of [cx, cy, w, h] (center format)
    probs: array of probabilities
    threshold: two boxes are considered overlapping if their IOU is largher than
        this threshold
    form: 'center' or 'diagonal'
  Returns:
    keep: array of True or False.
  """

  order = probs.argsort()[::-1]
  keep = [True]*len(order)

  for i in range(len(order)-1):
    ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]])
    for j, ov in enumerate(ovps):
      if ov > threshold:
        keep[order[j+i+1]] = False
  return keep

def batch_iou(boxes, box):
  """Compute the Intersection-Over-Union of a batch of boxes with another
  box.
  Args:
    box1: 2D array of [cx, cy, width, height].
    box2: a single array of [cx, cy, width, height]
  Returns:
    ious: array of a float number in range [0, 1].
  """
  lr = np.maximum(
      np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
      np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
      0
  )
  tb = np.maximum(
      np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
      np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
      0
  )
  inter = lr*tb
  union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
  return inter/union