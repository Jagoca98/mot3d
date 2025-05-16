import numpy as np
from utils.object import Object, ObjectBuilder
from utils.mot import MOT

if __name__ == "__main__":
    # Initialize the MOT system
    mot_system = MOT()

    # Create random object to add to the tracked list
    object_1 = ObjectBuilder.create(
        class_name="car",
        initial_state=np.ones(12)
    )

    object_2 = ObjectBuilder.create(
        class_name="pedestrian",
        initial_state=np.ones(12)
    )

    object_3 = ObjectBuilder.create(
        class_name="bicycle",
        initial_state=np.ones(12)*2
    )

    # List of tracked objects
    objects = []

    objects.append(object_1)
    objects.append(object_2)
    objects.append(object_3)    

    for i in range(10):

        # Update the tracked list with the new objects and perform the prediction
        obj_id = mot_system.update_tracked_list(objects)

        mot_system.predict_all()

        # print(f'Iteration {i}:')
        # print(f'Tracked: {mot_system.tracked_objects}')
        # print(f'Non Tracked: {mot_system.non_tracked_objects}')

    # Empty the objects list
    objects = []

    for i in range(15):

        # Add a new tracked object and perform the update
        obj_id = mot_system.update_tracked_list(objects)

        mot_system.predict_all()

        if len(mot_system.tracked_objects) > 0:
            print(type(mot_system.tracked_objects[obj_id].state))
            print(mot_system.tracked_objects[obj_id].state)

        
            