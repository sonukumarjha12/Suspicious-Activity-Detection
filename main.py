from Detector import Detector
import os

def main():
    # Use None or an empty string for videoPath if using the webcam
    videoPath = None  # or ""

    # Paths to model files
    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    # Initialize Detector with the necessary paths
    detector = Detector(videoPath, configPath, modelPath, classesPath)

    # Pass use_webcam=True to use the webcam
    detector.onVideo(use_webcam=True)

if __name__ == '__main__':
    main()
