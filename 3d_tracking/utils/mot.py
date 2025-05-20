import numpy as np

from .object import ObjectBuilder
from .ukf import BasicUKF
from .association.hungarian_association import HungarianAssociation


class MOT:
    """
    A class to handle Multiple Object Tracking (MOT) using a Kalman filter for each object.
    """
    
    def __init__(self):
        """
        Initialize the MOT system.
        """
        self.tracked_objects = {}
        self.non_tracked_objects = {}
        self.measurements = {}

        # next id
        self.next_id = 0

        # Appearance and disappearance thresholds
        self.appearance_threshold = 5
        self.disappearance_threshold = 5

        # Score threshold for the association
        self.score_threshold = 0.0

        # Association strategy
        self.association_strategy = HungarianAssociation(distance_threshold=1.0)


    def add_tracker(self, detection) -> int:
        """
        Add a new tracked object to the MOT system.
        Args:
            detection (Object): The detected object to be added.
        Returns:
            int: The ID of the newly added tracked object.
        """        
        detection.id = self.next_id
        self.next_id += 1
        self.non_tracked_objects[detection.id] = detection
        return detection.id
    
    def update_tracked_list(self, detections) -> bool:
        """
        Update the list of tracked objects based on appearance and disappearance counters.
        Promote objects that have been seen enough times to be tracked.
        Demote objects that have not been seen enough times to be removed to not tracked.
        """

        # Remove detections objects which low confidence score
        detections = [detection for detection in detections if detection.score > self.score_threshold]

        # First associate the tracked objects with the potential candidates
        association_result_1 = self.association_strategy.associate(
            tracked_objects=self.tracked_objects,
            detections=detections
        )

        # For each association, update the tracked object
        for obj_id, detection in association_result_1["associations"].items():
            pose = detection.get_pose()
            size = detection.size
            new_meassurements = np.append(pose, size)
            self.tracked_objects[obj_id].update(new_meassurements)
            self.tracked_objects[obj_id].on_detected()


        # If there are unassociated detections, try to associate them with non tracked objects
        association_result_2 = self.association_strategy.associate(
            tracked_objects=self.non_tracked_objects,
            detections=association_result_1["unassociated_detections"]
        )

        # For each new association, update the non tracked objects
        for obj_id, detection in association_result_2["associations"].items():
            pose = detection.get_pose()
            size = detection.size
            new_meassurements = np.append(pose, size)
            self.non_tracked_objects[obj_id].update(new_meassurements)
            self.non_tracked_objects[obj_id].on_detected()


        # Demote objects that have not been seen enough times from tracked to non tracked
        associated_ids = set(association_result_1["associations"].keys())
        to_downgrade = []
        for obj_id in list(self.tracked_objects.keys()):
            if obj_id not in associated_ids:
                obj = self.tracked_objects[obj_id]
                obj.on_missed()
                if obj.disappearance_counter > self.disappearance_threshold:
                    to_downgrade.append(obj_id)

        for obj_id in to_downgrade:
            obj = self.tracked_objects[obj_id]
            # Reset the appearance and disappearance counters to avoid removing the object
            # from the non tracked list immediately
            obj.appearance_counter = 0
            obj.disappearance_counter = 0
            self.non_tracked_objects[obj_id] = obj
            del self.tracked_objects[obj_id]


        # Promote objects that have been seen enough times from non tracked to tracked
        # and remove objects that have not been seen for a while from non tracked
        associated_non_ids = set(association_result_2["associations"].keys())
        to_promote = []
        to_remove = []
        for obj_id in list(self.non_tracked_objects.keys()):
            obj = self.non_tracked_objects[obj_id]
            if obj_id in associated_non_ids:
                # Was matched, already updated with on_detected() above
                pass
            else:
                # Not matched, update
                obj.on_missed()

            # Check if the object should be promoted or removed
            if obj.appearance_counter > self.appearance_threshold:
                to_promote.append(obj_id)
            elif obj.disappearance_counter > self.disappearance_threshold:
                to_remove.append(obj_id)


        # Promote objects that have been seen enough times to be tracked
        for obj_id in to_promote:
            self.tracked_objects[obj_id] = self.non_tracked_objects[obj_id]
            self.tracked_objects[obj_id].disappearance_counter = 0
            del self.non_tracked_objects[obj_id]

        # Remove objects that have not been seen for a while
        for obj_id in to_remove:
            del self.non_tracked_objects[obj_id]

        # Create a new non tracked object for each unassociated detection
        for detection in association_result_2["unassociated_detections"]:
            obj_id = self.add_tracker(detection)
            self.non_tracked_objects[obj_id].appearance_counter += 1
            self.non_tracked_objects[obj_id].disappearance_counter = 0

        return True
    

    def predict_all(self, dt: float = -1, **kwargs) -> None:
        """
        Predict the next state of all tracked objects and non tracked objects.
        Args:
            dt (float): The time step for prediction. If -1, use the default dt of the object.
        """
        # Predict the next state of all tracked objects
        for obj in self.tracked_objects.values():
            obj.predict(dt=dt, **kwargs)

        # Predict the next state of all non tracked objects
        for obj in self.non_tracked_objects.values():
            obj.predict(dt=dt, **kwargs)


        
