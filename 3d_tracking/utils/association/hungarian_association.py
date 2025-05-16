from typing import Dict, List, Any
from scipy.optimize import linear_sum_assignment
import numpy as np
from .basic_association import AssociationStrategy
from ..object import Object

class HungarianAssociation(AssociationStrategy):
    """
    Hungarian Algorithm-based association strategy.
    """

    def __init__(self, distance_threshold: float = 1.0):
        self.distance_threshold = distance_threshold

    def associate(self, tracked_objects: List[Object], detections: List[Object]) -> Dict[str, Any]:
        tracked_ids = list(tracked_objects.keys())

        # Get the poses of tracked objects and detections
        tracked_poses = [tracked_objects[obj_id].get_pose()[:3] for obj_id in tracked_ids]
        detections_poses = [detection.get_pose()[:3] for detection in detections]

        # If there are no tracked objects or detections, return empty associations
        if not tracked_poses or not detections:
            return {
                "associations": {},
                "unassociated_detections": detections,
                "unassociated_tracks": tracked_objects
            }

        cost_matrix = np.linalg.norm(
            np.expand_dims(tracked_poses, axis=1) - np.expand_dims(detections_poses, axis=0), axis=2
        )

        row_inds, col_inds = linear_sum_assignment(cost_matrix)

        associations = {}
        unassociated_tracks = set(tracked_ids)
        unassociated_detections = set(range(len(detections)))

        for row, col in zip(row_inds, col_inds):
            if cost_matrix[row, col] < self.distance_threshold:
                obj_id = tracked_ids[row]
                associations[obj_id] = detections[col]
                unassociated_tracks.discard(obj_id)
                unassociated_detections.discard(col)

        return {
            "associations": associations,
            "unassociated_detections": [detections[i] for i in unassociated_detections],
            "unassociated_tracks": list(unassociated_tracks)
        }
