import numpy as np
import cv2
from numba import njit
from typing import Tuple
import random

from utils.object import Object


class BirdView:
    def __init__(self, width=1920*3, height=1080*3) -> None:
        self.width = width
        self.height = height
        self.radius = 1
        self.crop = -100
        self.cell_size = 0.04
        self.lidarColor = (0, 0, 0)
        self.distanceColor = (109, 177, 245)

        # Pre-render background with distances
        self.background = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        self._drawDistances(self.background, radious=10, color=self.distanceColor)

        # Main image buffer
        self.bvImage = self.background.copy()

    def clear(self) -> None:
        self.bvImage = self.background.copy()

    def getBirdView(self) -> np.ndarray:
        return self.bvImage

    def drawPointCloud(self, points: np.ndarray) -> None:
        self._draw_points_on_image_only(self.bvImage, points, self.crop, self.width, self.height, self.cell_size, self.lidarColor, self.radius)

    def drawObjects(self, objects: dict[int, Object], thickness=2) -> None:
        for obj in objects.values():
            x, y, z, yaw = obj.get_pose()
            _, w, l = obj.size  # [h, w, l]

            tracking_data = [x, y, l, w, yaw]

            center_bev = self._project_single_point((tracking_data[0], tracking_data[1]))
            if center_bev is None:
                continue

            size = (tracking_data[2] / self.cell_size, tracking_data[3] / self.cell_size)
            angle = -(tracking_data[4] * 180 / np.pi)

            rect = ((center_bev[0], center_bev[1]), size, angle)
            box = cv2.boxPoints(rect)
            color = self._get_color(obj.id)

            cv2.polylines(self.bvImage, [box.astype(np.int32)], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    @staticmethod
    @njit
    def _draw_points_on_image_only(bvImage: np.ndarray, points: np.ndarray, crop: float, width: int, height: int, cell_size: float, lidarColor: Tuple[int, int, int], radius: int) -> None:
        for point_id in range(len(points)):
            point = points[point_id]
            if crop <= point[0] <= 200.0:
                p_x = int(point[0] / cell_size) + int(width / 2)
                p_y = int(-point[1] / cell_size) + int(height / 2)
                if radius <= p_x < bvImage.shape[1] - radius and radius <= p_y < bvImage.shape[0] - radius:
                    bvImage[p_y, p_x] = lidarColor
                    bvImage[p_y + 1, p_x] = lidarColor
                    bvImage[p_y - 1, p_x] = lidarColor
                    bvImage[p_y, p_x + 1] = lidarColor
                    bvImage[p_y, p_x - 1] = lidarColor

    def _project_single_point(self, point: Tuple[float, float]) -> Tuple[int, int] | None:
        x, y = point
        if self.crop <= x <= 200.0:
            px = int(x / self.cell_size) + int(self.width / 2)
            py = int(-y / self.cell_size) + int(self.height / 2)
            return (px, py)
        return None

    def _get_color(self, obj_id: int) -> Tuple[int, int, int]:
        rng = random.Random(obj_id)
        return (
            rng.randint(0, 255),
            rng.randint(0, 255),
            rng.randint(0, 255)
        )

    def _drawDistances(self, image: np.ndarray, radious=10, color=(109, 177, 245)) -> None:
        for i in range(0, 160, radious):
            cv2.circle(image, (self.width // 2, self.height // 2), int(i / self.cell_size), color, thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(image, str(i), (self.width // 2 + int(i / self.cell_size), self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
if __name__ == "__main__":
    pass
