import numpy as np

class BinaryPCD:
    """
    A class to handle binary PCD files.
    """

    def __init__(self):
        """
        Initialize the BinaryPCD object with a filename.
        :param filename: The name of the binary PCD file.
        """

    def read(self, filename) -> np.ndarray:
        """
        Read the binary PCD file and store the points and header.
        :param path: The path to the binary PCD file.
        """
        point_dtype = np.dtype([
                ('x', np.float32), 
                ('y', np.float32), 
                ('z', np.float32),
                #('intensity', np.uint8), 
            ])
        
        # Read the binary file and interpret it as an array of points
        scan = np.asarray(np.fromfile(filename, dtype=point_dtype).tolist())
        points = np.zeros((scan.shape[0],4))
        points[:,0:3] = scan[:,:3]

        return points
