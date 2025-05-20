import numpy as np

from .ukf import BasicUKF

class Object:
    """
    A class to represent a tracked object in 3D space.
    """
    
    def __init__(self, id: int, class_name, initial_state: np.array, score: float = 1.0, dt: float = 0.1):
        """
        Initialize the tracked object with an ID and initial state.
        """
        self.id = id
        self.class_name = class_name
        self.dt = dt
        self.ukf = BasicUKF(dt=self.dt, x_init=initial_state)
        self.appearance_counter = 0
        self.disappearance_counter = 0
        self.score = score

        # Initialize the state of the object
        self.state = initial_state

    @property
    def state(self):
        return self.ukf.ukf.x

    @state.setter
    def state(self, value):
        self.ukf.ukf.x = value

    @property
    def size(self):
        h, w, l = self.ukf.ukf.x[12], self.ukf.ukf.x[13], self.ukf.ukf.x[14]
        return np.array([h, w, l])
    
    @size.setter
    def size(self, value):
        if len(value) != 3:
            raise ValueError("Size must be a 3D vector [h, w, l]")
        self.ukf.ukf.x[12] = value[0]
        self.ukf.ukf.x[13] = value[1]
        self.ukf.ukf.x[14] = value[2]
    
    def predict(self, dt: float = -1, **kwargs) -> None:
        """
        Predict the next state of the tracked object.
        """
        # If df is not provided, use the default dt
        if dt == -1:
            dt = self.dt
        self.ukf.ukf.predict(dt=dt, **kwargs)
    
    def update(self, measurement: np.array, **kwargs) -> None:
        """
        Update the state of the tracked object with a new measurement.
        Args:
            measurement (np.array): The new measurement to update the state with [x, y, z, yaw].
            **kwargs: Additional keyword arguments for the update method.
        """
        self.ukf.ukf.update(measurement, **kwargs)

    def on_detected(self):
        self.appearance_counter += 1
        self.disappearance_counter = 0

    def on_missed(self):
        self.appearance_counter = 0
        self.disappearance_counter += 1


    def set_id(self, id: int) -> None:
        """
        Set the ID of the tracked object.
        Args:
            id (int): The ID to assign to the object.
        """
        self.id = id
    
    def get_pose(self) -> np.array:
        """
        Get the pose of the tracked object.
        Returns:
            np.array: The pose of the object in 3D space [x, y, z, yaw].
        """
        return self.ukf.get_pose()
    
    def get_size(self) -> np.array:
        """
        Get the size of the tracked object.
        Returns:
            np.array: The size of the object in 3D space [h, w, l].
        """
        h = self.ukf.ukf.x[12]
        w = self.ukf.ukf.x[13]
        l = self.ukf.ukf.x[14]
        return np.array([h, w, l])
        
    def get_state(self) -> np.array:
        return self.ukf.ukf.x

    def get_covariance(self) -> np.array:
        return self.ukf.P

    # Printing methods 
    def __repr__(self):
        return f"Object(id={self.id}, class_name={self.class_name})"
    
class ObjectBuilder:
    """
    A Builder class for creating Object instances with automatic unique IDs.
    """

    _next_id = 0

    @classmethod
    def create(cls, class_name, initial_state: np.array, score: float = 1.0, dt: float = 0.1) -> Object:
        obj = Object(id=cls._next_id, 
                     class_name=class_name, 
                     initial_state=initial_state, 
                     score=score,
                     dt=dt)
        cls._next_id += 1
        return obj