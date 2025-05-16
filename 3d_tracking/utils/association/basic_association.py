from typing import Dict, List, Any

class AssociationStrategy:
    """
    Interface for an association strategy between tracked objects and new detections.
    """

    def associate(self, tracked_objects: Dict[int, Any], detections: List[Any]) -> Dict[str, Any]:
        raise NotImplementedError("Association strategy must implement 'associate' method.")