import numpy as np
import os
from typing import List

from .object import ObjectBuilder, Object

class DetectionHandler:
    """
    A class to handle the detection of objects in a 3D space.
    """

    def __init__(self, rootDir: str = None):
        """
        Initialize the detection handler with a path to the data.
        """
        self.rootDir = rootDir
        self.filePaths = []

        self.load_frames_path()
    

    def load_frames_path(self):
        """
        Load the frames path from the root directory.
        """
        if self.rootDir is None:
            raise ValueError("Root directory is not set.")
        
        # Load all filepaths and append to an array from the directory if it ends with .txt
        self.filePaths_tmp = []
        self.filePaths_tmp = [f for f in os.listdir(self.rootDir) if f.endswith('.txt')]
        self.filePaths_tmp.sort()

        # Create the full path
        self.filePaths = [os.path.join(self.rootDir, f) for f in self.filePaths_tmp]


    def deserialize_detections(self, filePath: str) -> List[Object]:
        """
        Deserialize KITTI-style detections from a text file.

        Args:
            filePath (str): The path to the detection file.

        Returns:
            List[Object]: List of parsed Object instances.
        """
        objects = []

        with open(filePath, 'r') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 16:
                    print(f"Skipping line {line_num}: too short")
                    continue

                class_name = parts[2]
                x = float(parts[13])
                y = float(parts[14])
                z = float(parts[15])
                yaw = float(parts[16])

                initial_state = np.array([x, 0, 0, y, 0, 0, z, 0, 0, yaw, 0, 0])
                obj = ObjectBuilder.create(class_name, initial_state)
                objects.append(obj)

        return objects


