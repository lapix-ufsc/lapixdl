from __future__ import annotations

from copy import copy
from dataclasses import dataclass

from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon


@dataclass
class BBox:
    """Bounding Box data structure

    Attributes:
        upper_left_x (int): Upper left X position of the Bounding Box.
        upper_left_y (int): Upper left Y position of the Bounding Box.
        width (int): Width of the Bounding Box.
        height (int): Height of the Bounding Box.
        cls (int): Bounding Box class index.
        score (Optional[float]): Bounding Box prediction score.
    """
    upper_left_x: int
    upper_left_y: int
    width: int
    height: int
    cls: int
    score: float | None = None

    def __post_init__(self):
        if self.upper_left_x < 0 or self.upper_left_y < 0:
            raise ValueError(f'The upper left (x, y) should be positive values. Got ({self.upper_left_x}, {self.upper_left_y})')

        if self.width <= 0:
            raise ValueError(f'The width should be bigger than zero. Got {self.width}')

        if self.height <= 0:
            raise ValueError(f'The height should be bigger than zero. Got {self.height}')

    @property
    def upper_left_point(self) -> tuple[int, int]:
        """Tuple[int, int]: (X,Y) of the upper left point of the Bounding Box."""
        return (
            self.upper_left_x,
            self.upper_left_y
        )

    @property
    def upper_right_point(self) -> tuple[int, int]:
        """Tuple[int, int]: (X,Y) of the upper right point of the Bounding Box."""
        return (
            self.upper_left_x + self.width - 1,
            self.upper_left_y
        )

    @property
    def bottom_right_point(self) -> tuple[int, int]:
        """Tuple[int, int]: (X,Y) of the bottom right point of the Bounding Box."""
        return (
            self.upper_left_x + self.width - 1,
            self.upper_left_y + self.height - 1
        )

    @property
    def bottom_left_point(self) -> tuple[int, int]:
        """Tuple[int, int]: (X,Y) of the bottom left point of the Bounding Box."""
        return (
            self.upper_left_x,
            self.upper_left_y + self.height - 1
        )

    @property
    def center_point(self) -> tuple[int, int]:
        """Tuple[int, int]: (X,Y) of the center point of the Bounding Box."""
        return (
            self.upper_left_x + ((self.width - 1) // 2),
            self.upper_left_y + ((self.height - 1) // 2)
        )

    @property
    def area(self) -> int:
        """int: Area of the Bounding Box."""
        return self.width * self.height

    @property
    def coords(self) -> tuple[tuple[int, int], ...]:
        """Tuple[Tuple[int, int], ...]: A tuple with each point of the Bounding Box."""
        return (
            self.upper_left_point,
            self.upper_right_point,
            self.bottom_right_point,
            self.bottom_left_point,
        )

    @property
    def xy(self) -> tuple[list[int], list[int]]:
        """Tuple[list[int], list[int]]: A tuple with lists of coords for each x and y"""
        _x, _y = zip(*self.coords)
        return (list(_x), list(_y))

    @property
    def slice_x(self) -> slice:
        """slice: A slice between upper_left_x and upper_right_x"""
        return slice(self.upper_left_x, self.upper_right_point[0])

    @property
    def slice_y(self) -> slice:
        """slice: A slice between upper_left_y and bottom_left_y"""
        return slice(self.upper_left_y, self.bottom_left_point[1])

    def intersection_area_with(self: BBox, bbox: BBox) -> int:
        """Calculates the intersection area with another bbox

        Args:
            self (BBox): This bbox
            bbox (BBox): Bbox to intersect with

        Returns:
            int: The intersection area with the bbox
        """

        # Gets each box upper left and bottom right coordinates
        (upr_lft_x_a, upr_lft_y_a) = self.upper_left_point
        (btm_rgt_x_a, btm_rgt_y_a) = self.bottom_right_point

        (upr_lft_x_b, upr_lft_y_b) = bbox.upper_left_point
        (btm_rgt_x_b, btm_rgt_y_b) = bbox.bottom_right_point

        # Calculates the intersection box upper left and bottom right coordinates
        (upr_lft_x_intersect, upr_lft_y_intersect) = (
            max(upr_lft_x_a, upr_lft_x_b), max(upr_lft_y_a, upr_lft_y_b))
        (btm_rgt_x_intersect, btm_rgt_y_intersect) = (
            min(btm_rgt_x_a, btm_rgt_x_b), min(btm_rgt_y_a, btm_rgt_y_b))

        # Calculates the height and width of the intersection box
        (w_intersect, h_intersect) = (btm_rgt_x_intersect - upr_lft_x_intersect + 1,
                                      btm_rgt_y_intersect - upr_lft_y_intersect + 1)

        # If H or W <= 0, there is no intersection
        if (w_intersect <= 0) or (h_intersect <= 0):
            return 0

        return w_intersect * h_intersect

    def union_area_with(
            self: BBox,
            bbox: BBox,
            intersection_area: int | None = None
    ) -> int:
        """Calculates the union area with another bbox

        Args:
            self (BBox): This bbox
            bbox (BBox): Bbox to union with
            intersection_area (Optional[int], optional): The intersection area between this and bbox. Defaults to None.

        Returns:
            int: The union area with the bbox
        """

        return self.area + bbox.area - (intersection_area or self.intersection_area_with(bbox))

    def to_polygon(self) -> Polygon:
        """Polygon: A shapely polygon from the coordinates of the bbox"""
        return Polygon(self.coords)


@dataclass
class Annotation:
    geometry: Polygon | MultiPolygon
    category_id: int
    iscrowd: int = 0

    @property
    def bbox(self) -> BBox:
        return bounds_to_bbox(self.geometry.bounds, self.category_id)

    @property
    def _geo_type(self) -> str:
        return self.geometry.geom_type

    @property
    def xywh_bbox(self) -> list[int]:
        bbox = self.bbox
        return [bbox.upper_left_x, bbox.upper_left_y, bbox.width, bbox.height]

    def __iter__(self) -> Annotation:
        self._idx = 0

        if self._geo_type == 'MultiPolygon':
            self._geometries = list(self.geometry.geoms)
        elif self._geo_type == 'Polygon':
            self._geometries = [self.geometry]
        else:
            raise TypeError(f'Unexpected geometry type (`{self._geo_type}`) - expected `MultiPolygon` or `Polygon`')

        return self

    def __next__(self) -> Polygon:
        if self._idx < len(self._geometries):
            out = self._geometries[self._idx]
            self._idx += 1
            return out
        else:
            raise StopIteration

    def copy(self) -> Annotation:
        return copy(self)


def bounds_to_bbox(bounds: tuple[float], category_id: int) -> BBox:
    """Generate a BBox from a tuple of bounds

    Args:
        bounds (tuple[float]): A tuple of floats in the following order
    (min_x, min_y, max_x, max_y)
        category_id: A category id fot the BBox

    Returns:
        BBox: An BBox based on the bounds
    """
    b = tuple(int(i) for i in bounds)
    min_x, min_y, max_x, max_y = b
    return BBox(upper_left_x=min_x,
                upper_left_y=min_y,
                width=max_x - min_x,
                height=max_y - min_y,
                cls=category_id)
