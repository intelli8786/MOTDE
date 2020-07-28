import os
import cv2
import numpy as np

benchmarkPath = r"../Benchmark_Set"
weightsPath = r"../Weights"

from tracker.mot_tracker import OnlineTracker

from ObjectDetector.YOLOv2.customize.YOLOv2 import YOLOv2
yolov2 = YOLOv2(threshold=0.1, gpu=0.8,
                model=r"./ObjectDetector/YOLOv2/configs/yolov2-voc.cfg",
                labels=r"./ObjectDetector/YOLOv2/configs/labels.txt",
                weight=os.path.join(weightsPath, "yolov2-voc.weights"))

from Embedder.FaceNet.customize.FaceNet import FaceNet
facenet = FaceNet(model=os.path.join(weightsPath, "20170512-110547.pb"))

from ImageTransfer.SRGAN.customize.SRGAN import SRGAN
srgan = SRGAN(os.path.join(weightsPath, "SRGAN_G399.npz"))

from ObjectDetector.SSD.SSD_FaceDetection.customize.SSD import SSD
ssd = SSD(threshold=0.5, model=os.path.join(weightsPath, 'frozen_inference_graph_face.pb'))


def evaluation(challenges=[], resultDir="./results", configs={}, visualize=False):

    if not os.path.exists(resultDir):
        os.makedirs(resultDir)

    with open(os.path.join(resultDir, "info.txt"), 'w') as info:
        for key in sorted(configs.keys()):
            info.write(key + " : " + str(configs[key]) + "\r\n")

    motdt = OnlineTracker(models={"personDetect": yolov2, "faceEmbed": facenet, "superResolution": srgan, "faceDetect": ssd}, configs=configs)

    for challenge in challenges:
        motdt.clear()
        evalName = challenge.split('/')[-2]
        with open(os.path.join(resultDir, evalName + ".txt"), 'w') as result:
            imageNames = os.listdir(challenge)
            imageNames.sort()
            for imageName in imageNames:
                frame = cv2.imread(os.path.join(challenge, imageName), cv2.IMREAD_COLOR)

                # Tracking
                online_targets = motdt.update(frame)
                regions = []
                ids = []
                detects_face = []
                detects_person = []
                for t in online_targets:
                    regions.append(t.tlbr)
                    ids.append(t.track_id)


                # write evaluation result
                for region, id in zip(regions, ids):
                    print('%d,%d,%f,%f,%f,%f,1,-1,-1,-1' % (int(imageName.split(".")[0]), id, region[0], region[1], region[2] - region[0], region[3] - region[1]), file=result)


                # Visualize
                if visualize:
                    for region, id in zip(regions, ids):
                        cv2.rectangle(frame, (int(region[0]), int(region[1])), (int(region[2]), int(region[3])), (255, 255, 255), 2)
                        cv2.putText(frame, str(id), (int(region[0]), int(region[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
                    for detect in detects_person:
                        cv2.rectangle(frame, (int(detect[0]), int(detect[1])), (int(detect[2]), int(detect[3])), (255, 0, 0), 2)

                    for detect in detects_face:
                        cv2.rectangle(frame, (int(detect[0]), int(detect[1])), (int(detect[2]), int(detect[3])), (0, 0, 255), 2)

                    cv2.imshow('frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return

challenges = [os.path.join(benchmarkPath,r"Interaction-1/img1"),
              os.path.join(benchmarkPath,r"Interaction-2/img1"),
              os.path.join(benchmarkPath,r"Patrol/img1")]


# Baseline Model ------------------------------------------------------------------------------
#configs = {}
#configs["detect_threshold"] = 0.31
#resultDir = os.path.join("/root/based_MOTDT/results",
#            'detect_threshold %.2f' % configs["detect_threshold"]
#            )
#evaluation(challenges=challenges, resultDir=resultDir, configs=configs, visualize=False)
# ---------------------------------------------------------------------------------------------


# Add Redetection Method ----------------------------------------------------------------------
#configs = {}
#configs["detect_threshold"] = 0.31
#configs["redetect"] = True
#configs["redetect_target_min"] = 0.25
#configs["redetect_threshold"] = 0.61

#resultDir = os.path.join("/root/based_MOTDT/results",
#            'detect_threshold %.2f, ' % configs["detect_threshold"] + 
#            'redetect_target_min %.2f, ' % configs["redetect_target_min"] + 
#            'redetect_threshold %.2f' % configs["redetect_threshold"]
#            )
#evaluation(challenges=challenges, resultDir=resultDir, configs=configs, visualize=False)
# ---------------------------------------------------------------------------------------------


# Add Site Restoration Method -----------------------------------------------------------------
#configs = {}
#configs["detect_threshold"] = 0.31
#configs["recovery_body"] = True
#configs["recovery_body_method"] = "sr"
#configs["recovery_body_threshold"] = 24
#resultDir = os.path.join("/root/based_MOTDT/results",
#            'detect_threshold %.2f, ' % configs["detect_threshold"] + 
#            'recovery_body_threshold %s' % str(configs["recovery_body_threshold"]).zfill(3)
#            )
#evaluation(challenges=challenges, resultDir=resultDir, configs=configs, visualize=False)
# ---------------------------------------------------------------------------------------------


# Add Face Appearance Model -------------------------------------------------------------------
#configs = {}
#configs["detect_threshold"] = 0.31
#configs["faceRecognition_2"] = True
#configs["faceRecognition_2_method"] = "last"
#configs["faceRecognition_2_threshold"] = 0.65
#resultDir = os.path.join("/root/based_MOTDT/results",
#            'detect_threshold %.2f, ' % configs["detect_threshold"] + 
#            'faceRecognition_2_threshold %.2f' % configs["faceRecognition_2_threshold"]
#            )
#evaluation(challenges=challenges, resultDir=resultDir, configs=configs, visualize=False)
# ---------------------------------------------------------------------------------------------


# Final Model ---------------------------------------------------------------------------------
configs = {}
configs["detect_threshold"] = 0.31
configs["redetect"] = True
configs["redetect_target_min"] = 0.25
configs["redetect_threshold"] = 0.61
configs["recovery_body"] = True
configs["recovery_body_method"] = "sr"
configs["recovery_body_threshold"] = 24
configs["faceRecognition_2"] = True
configs["faceRecognition_2_method"] = "last"
configs["faceRecognition_2_threshold"] = 0.65

resultDir = os.path.join("../Benchmark_Results","based_MOTDT",
            'detect_threshold %.2f, ' % configs["detect_threshold"] + 
            'redetect_target_min %.2f, ' % configs["redetect_target_min"] + 
            'redetect_threshold %.2f, ' % configs["redetect_threshold"] +
            'recovery_body_threshold %s, ' % str(configs["recovery_body_threshold"]).zfill(3) +
            'faceRecognition_2_threshold %.2f' % configs["faceRecognition_2_threshold"]
            )
evaluation(challenges=challenges, resultDir=resultDir, configs=configs, visualize=False)
# ---------------------------------------------------------------------------------------------