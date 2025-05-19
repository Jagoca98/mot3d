import numpy as np
from tqdm import tqdm
import time
import cv2

from utils.object import Object, ObjectBuilder
from utils.mot import MOT
from utils.detection_handler import DetectionHandler
from utils.birdView import BirdView
from utils.binaryPCD import BinaryPCD

if __name__ == "__main__":
    # Initialize the MOT system
    mot_system = MOT()

    # Initialize the detection handler
    detection_handler = DetectionHandler(rootDir="/data/input/hw/nms_basic/")

    # Initalize the bird's-eye view drawer
    drawer = BirdView(width=1920*3, height=1080*3)

    # Create the BinaryPCD object
    binary_pcd = BinaryPCD()
    pcd_Path = "/data/input/hw/lidar_0/"

    # Create a video writer to save the output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('/data/output/video.avi', fourcc, 10.0, (1920*3, 1080*3))

    # for frame in detection_handler.filePaths:
    for frame in tqdm(detection_handler.filePaths):
        # Deserialize the detection from the file
        detections = detection_handler.deserialize_detections(frame)
        pointcloud = binary_pcd.read(pcd_Path + frame.split("/")[-1].replace(".txt", ".bin"))

        # Add the detected objects to the MOT system
        if (mot_system.update_tracked_list(detections=detections)):
            pass
        
        # Perform the prediction for all tracked objects
        mot_system.predict_all()

        # Clear the bird's-eye view image
        drawer.clear()

        # Draw the point cloud in the bird's-eye view
        drawer.drawPointCloud(points=pointcloud)

        if len(mot_system.tracked_objects) > 0:

            # Draw the tracked objects in the bird's-eye view
            drawer.drawObjects(objects=mot_system.tracked_objects, 
                               thickness=5)

        frame = drawer.getBirdView()        
        out.write(frame)
    
    # Release the video writer
    out.release()