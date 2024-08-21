import cv2
import numpy as np

class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        # Load the model
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        # Load the classes
        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
        self.classesList.insert(0, '__Background__')
        print(self.classesList)

    def onVideo(self, use_webcam=False):
        # Access the webcam if use_webcam is True, otherwise use the video file
        cap = cv2.VideoCapture(0) if use_webcam else cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print(f"Error opening {'webcam' if use_webcam else 'video file'}")
            return

        success, image = cap.read()
        while success:
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.4)
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))
            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

            if len(bboxIdx) != 0:
                for i in range(len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]

                    x, y, w, h = bbox

                    # Generate a random color based on the class label ID
                    color = (int(classLabelID * 37 % 255), int(classLabelID * 67 % 255), int(classLabelID * 97 % 255))

                    cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=2)
                    cv2.putText(image, f'{classLabel}: {classConfidence:.2f}', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Check if the detected object is a knife
                    if classLabel.lower() == "knife":
                        alert_text = "ALERT! Something violent detected. Please have a look."
                        cv2.putText(image, alert_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("result", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            success, image = cap.read()

        cap.release()
        cv2.destroyAllWindows()
