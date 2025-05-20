import numpy as np
from tqdm import tqdm
import time
import cv2

from utils.object import Object, ObjectBuilder
from utils.mot import MOT
from utils.detection_handler import DetectionHandler
from utils.birdView import BirdView
from utils.binaryPCD import BinaryPCD

def f_cv(x: np.array, dt: float) -> np.array:

    # Our model is:
    # x[0] = x
    # x[1] = x_dot
    # x[2] = x_dot_dot
    # x[3] = y
    # x[4] = y_dot
    # x[5] = y_dot_dot
    # x[6] = z
    # x[7] = z_dot
    # x[8] = z_dot_dot
    # x[9] = yaw
    # x[10] = yaw_dot
    # x[11] = yaw_dot_dot
    # x[12] = h
    # x[13] = w
    # x[14] = l

    # Initialize the output
    xout = x.copy()

    # Constant velocity model
    xout[0] += x[1] * dt
    xout[1] += 0
    xout[2] += 0
    xout[3] += x[4] * dt
    xout[4] += 0
    xout[5] += 0
    xout[6] += x[7] * dt
    xout[7] += 0
    xout[8] += 0
    xout[9] += x[10] * dt
    xout[10] += 0
    xout[11] += 0
    xout[12] += 0
    xout[13] += 0
    xout[14] += 0
    
    return xout

if __name__ == "__main__":
    # Initialize the MOT system
    mot_system = MOT()

    # Initialize the detection handler
    detection_handler = DetectionHandler(rootDir="/data/input/motion/nms_basic/")

    # Initalize the bird's-eye view drawer
    drawer = BirdView(width=1920*3, height=1080*3)

    # Create the BinaryPCD object
    binary_pcd = BinaryPCD()
    pcd_Path = "/data/input/motion/lidar_0/"

    # Create a video writer to save the output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('/data/output/motion_cv.avi', fourcc, 10.0, (1920*3, 1080*3))

    # for frame in detection_handler.filePaths:
    for frame in tqdm(detection_handler.filePaths):
        # Deserialize the detection from the file
        detections = detection_handler.deserialize_detections(frame)
        pointcloud = binary_pcd.read(pcd_Path + frame.split("/")[-1].replace(".txt", ".bin"))

        # Add the detected objects to the MOT system
        if (mot_system.update_tracked_list(detections=detections)):
            pass
        
        # Perform the prediction for all tracked objects
        mot_system.predict_all(dt=0.1, fx=f_cv)

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