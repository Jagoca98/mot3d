import numpy as np
from tqdm import tqdm

from utils.object import Object, ObjectBuilder
from utils.mot import MOT
from utils.detection_handler import DetectionHandler

if __name__ == "__main__":
    # Initialize the MOT system
    mot_system = MOT()

    # Initialize the detection handler
    detection_handler = DetectionHandler(rootDir="/data/input/hw/nms_basic/")

    # for frame in detection_handler.filePaths:
    for frame in tqdm(detection_handler.filePaths):
        # Deserialize the detection from the file
        detections = detection_handler.deserialize_detections(frame)

        # Add the detected objects to the MOT system
        if (mot_system.update_tracked_list(detections=detections)):
            pass
        
        # Perform the prediction for all tracked objects
        mot_system.predict_all()

        print(f"Num tracked: {len(mot_system.tracked_objects)}")
        print(f"Num non tracked: {len(mot_system.non_tracked_objects)}")
        

    # print(f"Detection files: {detection_handler.filePaths}")

    # # Create random object to add to the tracked list
    # object_1 = ObjectBuilder.create(
    #     class_name="car",
    #     initial_state=np.ones(12)
    # )

    # object_2 = ObjectBuilder.create(
    #     class_name="pedestrian",
    #     initial_state=np.ones(12)
    # )

    # object_3 = ObjectBuilder.create(
    #     class_name="bicycle",
    #     initial_state=np.ones(12)*2
    # )

    # # List of tracked objects
    # objects = []

    # objects.append(object_1)
    # objects.append(object_2)
    # objects.append(object_3)    

    # for i in range(10):

    #     # Update the tracked list with the new objects and perform the prediction
    #     obj_id = mot_system.update_tracked_list(objects)

    #     mot_system.predict_all()

    #     # print(f'Iteration {i}:')
    #     # print(f'Tracked: {mot_system.tracked_objects}')
    #     # print(f'Non Tracked: {mot_system.non_tracked_objects}')

    # # Empty the objects list
    # objects = []

    # for i in range(15):

    #     # Add a new tracked object and perform the update
    #     obj_id = mot_system.update_tracked_list(objects)

    #     mot_system.predict_all()

    #     if len(mot_system.tracked_objects) > 0:
    #         print(type(mot_system.tracked_objects[obj_id].state))
    #         print(mot_system.tracked_objects[obj_id].state)

        
            