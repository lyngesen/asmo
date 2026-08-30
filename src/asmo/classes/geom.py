from __future__ import annotations
import math
from shapely import MultiPolygon
from shapely.geometry import Polygon, LineString, Point, MultiLineString
import shapely.geometry
import queue as Q
import numpy as np
from shapely.geometry.base import BaseGeometry
import json


class Space:
    Z = 10**8
    geom = Polygon(((-Z, -Z), (-Z, Z), (Z, Z), (Z, -Z), (-Z, -Z)))
    q1 = Polygon(((0, 0), (0, Z), (Z, Z), (Z, 0), (0, 0)))
    q3 = Polygon(((0, 0), (0, -Z), (-Z, -Z), (-Z, 0), (0, 0)))


class AsmoPoint:

    # also allow for Point(1,2) as well as Point((1,2))

    def __init__(self, *args):
        if len(args) == 1:
            self.val = np.array(args[0])
        elif len(args) == 2:
            self.val = np.array(args)
        else:
            raise ValueError("Point must be initialized with two coordinates")

        self.geom: Point = Point(self.val)

    # def __init__(self, values: np.ndarray) -> None:
    # self.val : np.ndarray = values  # the vector containing point
    def __repr__(self):
        return tuple((float(vi) for vi in self.val)).__repr__()

    def __getitem__(self, i: int) -> float:
        return self.val[i]

    def __lt__(self, other: AsmoPoint) -> bool:
        # if both equal return false
        if self.val[0] == other.val[0] and self.val[1] == other.val[1]:
            return False
        # return True if self is less than or equal to other
        return self.val[0] <= other.val[0] and self.val[1] <= other.val[1]

    # add minkowski sum operator of AsmoPoint and Polygon
    def __add__(self, other: Polygon) -> BaseGeometry:
        # if the other one is a AsmoPoint, return the sum of the two points (class AsmoPoint)
        if isinstance(other, AsmoPoint):
            # return AsmoPoint(self.val[0] + other.val[0], self.val[1] + other.val[1])
            print(f"{self.val,other.val,self.val+other.val=}")
            return AsmoPoint(self.val + other.val)

        # define minkowski sum of point and Polygon
        elif isinstance(other, Polygon):
            result_coords = []
            for x, y in other.exterior.coords:
                result_coords.append((self.geom.x + x, self.geom.y + y))
            result_polygon = Polygon(result_coords)
            return result_polygon

    def __mul__(self, other: float) -> AsmoPoint:
        # return a new AsmoPoint object
        return AsmoPoint(self.val * other)

    def __neg__(self) -> AsmoPoint:
        # return a new AsmoPoint object
        return self.__mul__(-1)


class Rectangle:
    def __init__(self, ul: AsmoPoint, lr: AsmoPoint, integergap=False):
        if __debug__:
            # print(f"Creating rectangle with ul: {ul}, lr: {lr}")
            pass
        self.ul: AsmoPoint = ul
        self.lr: AsmoPoint = lr
        if integergap:
            # create a smaller rectangle by removing the integer gap
            self.ul = AsmoPoint((self.ul[0] + 1, self.ul[1] - 1))
            self.lr = AsmoPoint((lr[0] - 1, lr[1] + 1))

        if self.ul[0] < lr[0] and self.ul[1] > lr[1]:
            # Create rectangle from the two diagonal points
            self.geom = Polygon(
                [
                    (self.ul[0], self.ul[1]),
                    (self.lr[0], self.ul[1]),
                    (self.lr[0], self.lr[1]),
                    (self.ul[0], self.lr[1]),
                    (self.ul[0], self.ul[1]),
                ]
            )
        elif self.ul == self.lr:
            # If the two points are the same, create a point
            self.geom = Point(self.ul[0], self.ul[1])
            print("Warning: point1 and point2 are the same. Creating a point.")
        # if one point dominates the other
        elif self.ul[0] < lr[0] and self.ul[1] < lr[1]:  # ul dominates lr
            self.geom = Point()
        elif self.ul[0] > lr[0] and self.ul[1] > lr[1]:  # lr dominates ul
            self.geom = Point()
        else:
            self.geom = LineString((self.ul.geom, self.lr.geom))
            print(
                "Warning: point1 is not to the left of point2. Creating a LineString."
            )
            # raise ValueError("Point1 is not to the left of point2", self.ul, lr)
            # raise ValueError("Point1 is not to the left of point2")
            # self.geom = Polygon()
            # super().__init__()

    # add union, intersection, difference, etc. methods by calling self.geom.union etc
    def union(self, other: Rectangle) -> BaseGeometry:
        return self.geom.union(other.geom)

    def intersection(self, other: Rectangle) -> BaseGeometry:
        return self.geom.intersection(other.geom)

    def difference(self, other: Rectangle) -> BaseGeometry:
        return self.geom.difference(other.geom)

    def symmetric_difference(self, other: Rectangle) -> BaseGeometry:
        return self.geom.symmetric_difference(other.geom)

    def buffer(self, distance: float) -> BaseGeometry:
        return self.geom.buffer(distance)

    def get_upper_bound(self) -> Bound:
        """docstring for get_upper_bound"""
        return Bound(
            [
                AsmoPoint(-Space.Z, self.ul[1]),
                AsmoPoint(self.lr[0], self.ul[1]),
                AsmoPoint(self.lr[0], -Space.Z),
            ]
        )

    def get_lower_bound(self) -> Bound:
        """docstring for get_lower_bound"""
        return Bound(
            [
                AsmoPoint(self.ul[0], Space.Z),
                AsmoPoint(self.ul[0], self.lr[1]),
                AsmoPoint(Space.Z, self.lr[1]),
            ]
        )


class Line:
    """docstring for Line class"""

    def __init__(self, *args):
        if len(args) == 1:
            self.val = np.array(args[0])
        elif len(args) == 2:
            self.val = np.array(args)
        else:
            raise ValueError("Line must be initialized with two coordinates")
        for v in self.val:
            assert v.__class__.__name__.endswith(
                "AsmoPoint"
            ), f"Line must be initialized with AsmoPoint, got {v.__class__.__name__}"
        self.geom = LineString(self.val)

    # get item
    def __getitem__(self, i: int) -> AsmoPoint:
        return self.val[i]

    @property
    def ideal(self) -> type:
        x_min = min((self.val[0][0]), self.val[1][0])
        y_min = min((self.val[0][1]), self.val[1][1])
        return AsmoPoint(x_min, y_min)

    @property
    def nadir(self):
        x_max = max((self.val[0][0]), self.val[1][0])
        y_max = max((self.val[0][1]), self.val[1][1])
        return AsmoPoint(x_max, y_max)

    def __add__(self, other: Line | AsmoPoint) -> Line | Polygon:
        # if the other one is a AsmoPoint, return the sum of the two points (class AsmoPoint)
        if isinstance(other, AsmoPoint) or other.__class__.__name__.endswith(
            "AsmoPoint"
        ):
            # return AsmoPoint(self.val[0] + other.val[0], self.val[1] + other.val[1])
            # return Line(AsmoPoint(self[0].val + other.val ) ,  AsmoPoint(self[0].val + other.val))
            # return a new Line object also defined by two AsmoPoint objects
            return Line(
                AsmoPoint(self[0].val + other.val), AsmoPoint(self[1].val + other.val)
            )
        elif isinstance(other, Line) or other.__class__.__name__.endswith("Line"):
            # Should return the convex hull of the pairs of points from the two sets, ie Conv([self[0]+other[0],self[0]+other[1],..., self[1]+other[1]])
            coords = [
                self[0].val + other[0].val,
                self[0].val + other[1].val,
                self[1].val + other[1].val,
                self[1].val + other[0].val,
            ]

            # Compute centroid
            cx = sum(p[0] for p in coords) / len(coords)
            cy = sum(p[1] for p in coords) / len(coords)

            # Sort counter-clockwise around centroid (needed for the get_upper get_lower part)
            coords = sorted(coords, key=lambda p: -math.atan2(p[1] - cy, p[0] - cx))

            return Polygon(coords)

    def add_max(self, other: Line) -> Iterator[Line]:
        # Given two line segments, this function should
        pass

    def __mul__(self, other: float) -> Line:
        # return a new Line object
        return Line([AsmoPoint(self[0].val * other), AsmoPoint(self[1].val * other)])

    def __neg__(self) -> Line:
        # return a new Line object, negated
        return self.__mul__(-1)


class Bound:
    """docstring for Bound set"""

    # same type of class as Rectangle, but using the LineString geometry. Allow for creating Bound((1,2), (3,4)) as well as Bound([1,2], [3,4]) and Bound(((1,2),(2,3)))
    def __init__(self, *args):
        if len(args) == 0:
            self.coords = []
        elif len(args) == 1:
            if isinstance(args[0], (LineString)):
                self.coords = list(args[0].coords)
            else:
                self.coords = args[0]
        elif args[0].__class__.__name__.endswith("Solution"):
            self.coords = [p.val for p in args]
        else:
            self.coords = args

        # print(f"{self.coords=}")
        def is_asmo_point(p):
            return p.__class__.__name__.endswith("AsmoPoint")

        if self.coords:
            pass
            # self.coords = [(AsmoPoint(p) if not is_asmo_point(p) else p ) for p in self.coords]
            # assert self.coords[0].__class__.__name__.endswith("AsmoPoint"), f"Bound must be initialized with AsmoPoint, got {self.coords[0].__class__.__name__}"

        self.coords = sorted(
            self.coords, key=lambda p: (p[0], -p[1])
        )  # always sort lexico
        self.geom = LineString(self.coords)

    @staticmethod
    def from_MultiLineString(mls: MultiLineString) -> Bound:
        return Bound(mls.coords)

    def get_lines(self) -> list[Line]:
        # return a list of lines from the bound set
        lines = []
        for i in range(len(self.coords) - 1):
            lines.append(Line(self.coords[i], self.coords[i + 1]))
        return lines

    def __mul__(self, other: float) -> Bound:
        # return a new Bound object
        return Bound([AsmoPoint(p.val * other) for p in self.coords])

    def __neg__(self) -> Bound:
        # return a new Bound object
        return Bound([AsmoPoint(-p.val) for p in self.coords])

    def __add__(
        self, other: Bound | Polygon | AsmoPoint
    ) -> MultiPolygon | Polygon | Bound:
        # define minkowski sum of bound set and Polygon
        if isinstance(other, Polygon):
            result_coords = []
            for point in self.coords:
                for x, y in other.exterior.coords:
                    result_coords.append((point.geom.x + x, point.geom.y + y))
        # or if class name ends with "AsmoPoint"
        elif isinstance(other, AsmoPoint) or other.__class__.__name__.endswith(
            "AsmoPoint"
        ):
            result_coords = []
            for x, y in other.geom.coords:
                for point in self.coords:
                    result_coords.append(AsmoPoint(point.geom.x + x, point.geom.y + y))
            return Bound(result_coords)

        elif isinstance(other, Bound) or other.__class__.__name__.endswith("Bound"):
            # Compute the Minkowski sum of two Bound objects
            # The bound sets are composed of linestrings — a set of line segments
            # polygon = MultiPolygon() #empty polygon
            # polygon = None
            # for l1 in self.get_lines():
            #     for l2 in other.get_lines():
            #         # Compute the Minkowski sum of the two lines
            #         # result_coords.append(l1.geom.union(l2.geom))
            #         if polygon is None:
            #             polygon = l1+l2
            #         else:
            #             polygon.union(l1 + l2)
            #

            # return bound_plus_bound(self, other)

            return shapely.union_all(
                [l1 + l2 for l1 in self.get_lines() for l2 in other.get_lines()]
            )

            union = None
            for l1 in self.get_lines():
                for l2 in other.get_lines():
                    if union is None:
                        union = l1 + l2
                    else:
                        union = union.union(l1 + l2)

            return union

            return Polygon
            # polygons.append(l1.geom.union(l2.geom))

            # result_coords = []
            # for point1 in self.coords:
            #     for point2 in other.coords:
            #         result_coords.append((point1.geom.x + point2.geom.x, point1.geom.y + point2.geom.y))
        else:
            raise TypeError("Operand must be an instance of Bound")

        result_polygon = Polygon(result_coords)
        return result_polygon

    # create intersection with Rectangle, returning a subset of the bound

    def intersection(self, other: Rectangle) -> BaseGeometry:
        return self.geom.intersection(other.geom)

    def add_cone(self, quadrant, Z=10 * 1000) -> BaseGeometry:
        # add the cone Rpp to the bound set bounded by some large Z
        assert quadrant in [1, 2, 3, 4]
        if quadrant == 1:
            cone = Polygon([(0, 0), (Z, 0), (Z, Z), (0, Z), (0, 0)])
        elif quadrant == 2:
            cone = Polygon([(0, 0), (-Z, 0), (0, Z), (0, 0)])
        elif quadrant == 3:
            cone = Polygon([(0, 0), (-Z, 0), (0, -Z), (0, 0)])
        elif quadrant == 4:
            cone = Polygon([(0, 0), (Z, 0), (0, -Z), (0, 0)])
        # return minkowski sum of cone and bound.geom
        return self.__add__(cone)

    def dominated_by_space(self) -> Polygon:
        """returns the intersection of the space Z with the self MS R^2_\\leqq"""
        Z = Space.Z
        coords = list(sorted(self.geom.coords, key=lambda x: (x[0], -x[1])))
        lr = Point((coords[-1][0], -Z))
        ll = Point((-Z, -Z))
        ul = Point((-Z, coords[0][1]))
        return Polygon(coords + [lr, ll, ul])

    def dominates_space(self) -> Polygon:
        """returns the intersection of the space Z with the self MS R^2_\\geqq"""
        Z = Space.Z
        coords = list(sorted(self.geom.coords, key=lambda x: (x[0], -x[1])))
        lr = Point((coords[-1][1], Z))
        ur = Point((Z, Z))
        ul = Point((Z, coords[0][0]))
        return Polygon(coords + [ul, ur, lr])

    def _get_nadir_(self):
        minx, miny, maxx, maxy = self.geom.bounds
        return AsmoPoint(maxx, maxy)

    def _merge_bounds(self, other: Bound, direction="upper", debug=False):

        if __debug__ and False:
            self.save_json("../instances/tests/debug/_merge_bounds_self.json")
            other.save_json("../instances/tests/debug/_merge_bounds_other.json")

        Z = Space.Z
        sminx, sminy, smaxx, smaxy = self.geom.bounds
        ominx, ominy, omaxx, omaxy = other.geom.bounds
        minx = min(sminx, ominx)
        miny = min(sminy, ominy)
        maxx = max(smaxx, omaxx)
        maxy = max(smaxy, omaxy)

        if direction == "upper":
            # the upper bound is the ND set of the union og the two bounds + the cone
            A_self = Polygon(
                [y for y in self.geom.coords]
                + [
                    (maxx + Z, sminy),
                    (maxx + Z, maxy + Z),
                    (sminx, maxy + Z),
                ]
            )
            A_other = Polygon(
                [y for y in other.geom.coords]
                + [
                    (maxx + Z, ominy),
                    (maxx + Z, maxy + Z),
                    (ominx, maxy + Z),
                ]
            )

            assert A_self.is_valid, f"Invalid polygon: {A_self}"
            assert A_other.is_valid, f"Invalid polygon: {A_other}"

            union = A_self.union(A_other)

            if False:
                from classes.plotter import Plotter

                P = Plotter()
                P.plot(union, name="union")
                P.plot(A_self, name="A_self", color="blue")
                P.plot(A_other, name="A_other", color="red")
                coords = sorted(union.exterior.coords, key=lambda x: (x[0], -x[1]))
                new_lower_bound = Bound(
                    [AsmoPoint(y) for y in coords if y[0] <= maxx and y[1] <= maxy]
                )
                P.plot(new_lower_bound, color="green", linestyle="--")
                P.save("_merge_bound_debug.pdf")
        elif direction == "lower":
            # the lower bound is the ND set of the intersection og the two bounds + the cone
            # NOTE: This assumes that both are lower bounds, ie. any point of Y must be dominated by a point in L (unlike an upper bound only needs to not be domianted)
            A_self = Polygon(
                [y for y in self.geom.coords]
                + [
                    (smaxx, miny),
                    (maxx + Z, miny),
                    (maxx + Z, maxy + Z),
                    (minx, maxy + Z),
                    (minx, smaxy),
                ]
            )
            A_other = Polygon(
                [y for y in other.geom.coords]
                + [
                    (omaxx, miny),
                    (maxx + Z, miny),
                    (maxx + Z, maxy + Z),
                    (minx, maxy + Z),
                    (minx, omaxy),
                ]
            )

            assert A_self.is_valid, f"Invalid polygon: {A_self}"
            assert A_other.is_valid, f"Invalid polygon: {A_other}"
            union = A_self.intersection(A_other, grid_size=1)
        else:
            raise ValueError("Direction must be either 'upper' or 'lower'", direction)

        coords = sorted(union.exterior.coords, key=lambda x: (x[0], -x[1]))

        # return Bound induced by lower part of the polygon, ignoring the corner points ul, lr and nadir
        new_lower_bound = Bound(
            [AsmoPoint(y) for y in coords if y[0] <= maxx and y[1] <= maxy]
        )
        if debug:
            return new_lower_bound, A_self, A_other, union
        return new_lower_bound

    def merge_upper_bounds(self, other: Bound):
        """docstring for merge_upper_bounds"""
        return self._merge_bounds(other, direction="upper")

    def merge_lower_bounds(self, other: Bound, debug=False) -> Bound:
        """docstring for merge_lower_bounds"""
        return self._merge_bounds(other, direction="lower", debug=debug)

    def save_json(self, filename: str) -> None:
        # convert the 2d coords of self.coords to a list of tuples and save as file
        formattet_points = [tuple((float(vi) for vi in p.val)) for p in self.coords]
        D = {"coords": formattet_points}
        json.dump(D, open(filename, "w"), indent=4)

    @staticmethod
    def load_json(filename: str) -> Bound:
        # load the json file and convert to Bound
        D = json.load(open(filename, "r"))
        coords = [AsmoPoint(p) for p in D["coords"]]
        return Bound(coords)

    def extend_lower(self) -> Bound:
        # assert coords are sorted
        if __debug__:
            assert self.coords == sorted(
                self.coords, key=lambda p: (p[0], -p[1])
            ), "Bound coords must be sorted before extending"
        return Bound(
            [AsmoPoint(self.coords[0][0], Space.Z)]
            + list(self.coords)
            + [AsmoPoint(Space.Z, self.coords[-1][1])]
        )

    def extend_upper(self, Z: int = None) -> Bound:
        if Z is None:
            Z = Space.Z
        return Bound(
            [AsmoPoint(-Z, self.coords[0][1])]
            + list(self.coords)
            + [AsmoPoint(self.coords[-1][0], -Z)]
        )

    def without_dom_lines(self):
        # return an iterable containing all not horizontal and vertical lines
        lines = [
            l
            for l in self.get_lines()
            if not (l[0][0] == l[1][0] or l[0][1] == l[1][1])
        ]
        # return a Shapely MultiLineString object containing all lines
        return MultiLineString([l.geom for l in lines])


class SearchArea:
    """docstring for SearchRegion class"""

    # Define a searchArea, takes Polygon as input

    def __init__(self, polygon: Polygon | MultiPolygon) -> None:
        self.geom = polygon
        # assert self.geom.__class__.__name__.endswith("Polygon"), f"SearchArea must be initialized with Polygon, got {self.geom.__class__.__name__}"

    def N(self) -> Bound:
        """docstring for N, get lower envelope pareto fron of searcharea"""

    def __mul__(self, other: float) -> SearchArea:
        # return a new SearchArea object, scaled by a factor other
        return SearchArea(
            Polygon([np.array(coord) * other for coord in self.geom.exterior.coords])
        )

    def __neg__(self) -> SearchArea:
        # return a new SearchArea object, negated
        return self.__mul__(-1)

    def save_json(self, filename: str) -> None:

        if type(self.geom).__name__.endswith(
            "MultiPolygon"
        ) or self.geom.__class__.__name__.endswith("GeometryCollection"):
            # if the geometry is a MultiPolygon, convert to Polygon
            # geom_formatted = [tuple((float(vi) for vi in p.exterior.coords)) for p in self.geom.geoms]
            D = {"geoms": list(), "shapeType": self.geom.__class__.__name__}
            for geom in self.geom.geoms:
                assert geom.is_valid, f"Invalid polygon: {geom}"
                if geom.__class__.__name__.endswith("Polygon"):
                    d = {
                        "coords": [
                            tuple((float(vi) for vi in p)) for p in geom.exterior.coords
                        ],
                        "shapeType": "Polygon",
                    }
                elif geom.__class__.__name__.endswith("LineString"):
                    d = {
                        "coords": [tuple((float(vi) for vi in p)) for p in geom.coords],
                        "shapeType": "LineString",
                    }
                elif geom.__class__.__name__.endswith("Point"):
                    print(tuple(geom.coords))
                    d = {
                        "coords": tuple((float(vi) for vi in geom.coords[0])),
                        "shapeType": "Point",
                    }
                else:
                    raise ValueError(
                        f"Unsupported geometry type: {geom.__class__.__name__}"
                    )
                D["geoms"].append(d)
        else:
            # convert the 2d coords of self.coords to a list of tuples and save as file
            formattet_points = [
                tuple((float(vi) for vi in p)) for p in self.geom.exterior.coords
            ]
            D = {"coords": formattet_points, "shapeType": self.geom.__class__.__name__}

        json.dump(D, open(filename, "w"), indent=4)

    def load_json(filename: str) -> SearchArea:
        # load the json file and convert to SearchArea
        D = json.load(open(filename, "r"))
        if D["shapeType"] == "MultiPolygon":
            coords = []
            for geom in D["geoms"]:
                assert (
                    geom["shapeType"] == "Polygon"
                ), f"Expected Polygon, got {geom['shapeType']}"
                coords.append([AsmoPoint(p) for p in geom["coords"]])
            coords = [Polygon(c) for c in coords]
            return SearchArea(MultiPolygon(coords))
        else:
            coords = [p for p in D["coords"]]
        return SearchArea(Polygon(coords))

    @staticmethod
    def get_search_area_geom(L: Bound, U: Bound) -> BaseGeometry:
        """docstring for get_search_area_geom"""

        M = Space.Z
        U_dom = shapely.buffer(U.geom, M, single_sided=True)

        Rk = L.geom.union(U.geom).envelope

        Ak = (
            Rk.difference(U_dom)
            .intersection(shapely.buffer(L.geom, 400, single_sided=True))
            .intersection(Rk)
        )

        # Ak = shapely.buffer(L.geom, M/2, single_sided=True).intersection(shapely.buffer(U_dom, M/2, single_sided=True)).intersection(Space.geom)
        return Ak

    @staticmethod
    def from_bound_sets(n: "BnbNode", L: Bound, U: Bound):
        """docstring for from_bound_sets
        explanation of method found in docs/documentation/problem-BnbNode_check_if_empty.md
        """

        # manual approach
        assert (
            L.geom.geom_type == "LineString"
        ), f"Expected Linestring, got {L.geom.geom_type}"
        assert (
            U.geom.geom_type == "LineString"
        ), f"Expected Linestring, got {U.geom.geom_type}"
        # assert (
        #     n.R.geom.geom_type == "Polygon"
        # ), f"Expected Polygon, got {n.R.geom.geom_type}"
        # #
        # L_ref = Point(()) # L + R^2_\geqq
        # U_ref = Point(()) # U + R^2_\leqq
        L_dominated = L.dominates_space()
        U_dominates = U.dominated_by_space()

        # assert geoms are valid
        assert L_dominated.is_valid, f"Invalid polygon: {L_dominated}"
        assert U_dominates.is_valid, f"Invalid polygon: {U_dominates}"
        # assert n.R.geom.is_valid, f"Invalid polygon: {n.R.geom}"

        return (
            L.dominates_space()
            .intersection(U.dominated_by_space())
            .intersection(n.R.geom.convex_hull)
        )  # added convex_hull

        # if not L.geom.intersects(n.R.geom):
        # return Polygon()
        Lk = L.intersection(n.R)
        # l1 = AsmoPoint(Lk.coords[0])
        # l2 = AsmoPoint(Lk.coords[-1])
        U_dom = shapely.buffer(U.geom, 100000, single_sided=True)
        # Rk = Rectangle(l1,l2)
        Rk = n.R

        Ak = Rk.geom.difference(U_dom).intersection(
            shapely.buffer(Lk, 400, single_sided=True)
        )

        return Ak

    @property
    def ul(self) -> AsmoPoint:
        minx, miny, maxx, maxy = self.geom.bounds
        return AsmoPoint((minx, maxy))

    @property
    def lr(self) -> AsmoPoint:
        minx, miny, maxx, maxy = self.geom.bounds
        return AsmoPoint((maxx, miny))

    @staticmethod
    def _get_polygon_lex_min(A: Polygon) -> tuple[AsmoPoint, AsmoPoint]:
        # sorted_coords = sorted(A.exterior.coords, key=lambda x: (-x[0], x[1]))
        # y_ul = sorted_coords[0]
        # y_lr = sorted_coords[-1]

        points = A.exterior.coords
        a_ul = min(points, key=lambda p: (p[0], -p[1]))
        a_lr = min(points, key=lambda p: (p[1], -p[0]))
        return (AsmoPoint(a_ul), AsmoPoint(a_lr))

    def get_lex_min(self) -> tuple[AsmoPoint, AsmoPoint]:
        """docstring for get_lex_min"""
        if self.geom.__class__.__name__.endswith("MultiPolygon"):
            points = sum([list(p.exterior.coords) for p in self.geom.geoms], [])

            a_ul = min(points, key=lambda p: (p[0], -p[1]))
            a_lr = min(points, key=lambda p: (p[1], -p[0]))
            return (AsmoPoint(a_ul), AsmoPoint(a_lr))
            # sorted_coords = sorted(all_coords, key=lambda x: (-x[0], x[1]))
            # y_ul = sorted_coords[0]
            # y_lr = sorted_coords[-1]
            # return (AsmoPoint(y_ul), AsmoPoint(y_lr))
        else:
            return self._get_polygon_lex_min(self.geom)

    @staticmethod
    def _get_part_of_polygon(A: Polygon, part: str) -> list[AsmoPoint]:
        assert part in [
            "upper",
            "lower",
        ], f"part must be either 'upper' or 'lower', got {part}"
        points = list(A.exterior.coords)
        if part == "upper":
            upper_left = min(points, key=lambda p: (p[0], -p[1]))
            lower_right = min(points, key=lambda p: (p[1], -p[0]))
        elif part == "lower":
            upper_left = min(points, key=lambda p: (p[1], -p[0]))
            lower_right = min(points, key=lambda p: (p[0], -p[1]))
        # Find the indices of the upper_left and lower_right points
        ul_index = points.index(upper_left)
        lr_index = points.index(lower_right)
        # If the upper_left point comes before the lower_right point in the list, return the points between them
        if ul_index < lr_index:
            return points[ul_index : lr_index + 1]
        else:
            return points[ul_index:] + points[: lr_index + 1]

        y_ul, y_lr = SearchArea._get_polygon_lex_min(A)
        i_lr = [AsmoPoint(y) for y in A.exterior.coords].index(y_lr)
        i_ul = [AsmoPoint(y) for y in A.exterior.coords].index(y_ul)
        # i_ul = list(A.exterior.coords).index(y_ul)
        return [
            AsmoPoint(y)
            for i, y in enumerate(A.exterior.coords)
            if i in range(i_ul, i_lr + 1)
        ]

    def get_bound(self, part: str) -> Bound:
        # create a decorator which saves the input and output of the function to a json file

        if __debug__:
            self.save_json("./instances/tests/debug/_get_bound.json")

        assert part in [
            "upper",
            "lower",
        ], f"part must be either 'upper' or 'lower', got {part}"
        bound_points = []
        if self.geom.__class__.__name__.endswith("GeometryCollection"):
            for geom in sorted(
                self.geom.geoms,
                key=lambda x: (x.centroid.x if part == "upper" else -x.centroid.x),
            ):  # sort the polygons first
                if geom.geom_type == "Polygon":
                    bound_points += self._get_part_of_polygon(geom, part=part)
                elif geom.geom_type == "Point":
                    bound_points.append(geom.coords[0])
                elif geom.geom_type == "LineString":
                    for point in geom.coords:
                        bound_points.append(point)
            # raise ValueError("SearchArea must be initialized with Polygon or MultiPolygon, got GeometryCollection")
        elif self.geom.__class__.__name__.endswith("MultiPolygon"):
            # get the upper bound of the search area
            for polygon in sorted(
                self.geom.geoms,
                key=lambda x: (x.centroid.x if part == "upper" else -x.centroid.x),
            ):  # sort the polygons first
                bound_points += self._get_part_of_polygon(polygon, part=part)
        else:
            bound_points += self._get_part_of_polygon(self.geom, part=part)
        # i_ul. The index of the upper left element in coords of A

        return Bound([AsmoPoint(y) for y in bound_points])

    def get_upper_bound(self) -> Bound:
        """docstring for get_upper_bound"""
        return self.get_bound("upper")

    def get_lower_bound(self) -> Bound:
        """docstring for get_lower_bound"""
        return self.get_bound("lower")


class BnbNode:  # delsøgeområde
    """This class implements a simple branch and bound node. This node will store information about the subproblem
    corresponding to the node
    """

    def __init__(self, p: Problem, hL: Bound, R: Rectangle):
        self.p: Problem = p  # problem associated with node (previous)
        self.hL: Bound = hL  # lower bound region associated witht the node
        self.R: Rectangle = R  # Rectangle containing the search area
        self.depth = 0  # The depth of the node in the branching tree. Useful when searching depth first/breadth first
        self.sortingKey = 0  # Can be used if nodes should be sorted based on this key

    def initializeNode(self, farther: "node", branchingVar: int, value: int):
        """
        This method initializes the node-object
        :param farther: a node corresponding to the farther node of the node that should be initialized
        :param branchingVar: the variable that is being branched on when creating this node
        :param value: value that the branching variable should be fixed to (must be either 0 (zero) or 1 (one))
        """
        self.depth = farther.depth + 1
        self.fixedToOne = list(farther.fixedToOne)
        self.fixedToZero = list(farther.fixedToZero)
        self.free = [i for i in farther.free if i != branchingVar]
        if value == 1:
            self.fixedToOne.append(branchingVar)
        elif value == 0:
            self.fixedToZero.append(branchingVar)
        else:
            raise ValueError(
                f"Branching value should be either zero or one. Yor provided {value}. Depth: {self.depth}"
            )

    def __lt__(self, other):
        """Less than comparison for the node-class"""
        return self.sortingKey < other.sortingKey


class BranchingTree:
    """
    Class implementing a branching tree structure for storing branching nodes
    """

    def __init__(self):
        self.T = Q.PriorityQueue()  # Priority queue used for storing branching nodes

    def getFrontSortingKey(self):
        """Method returns the sorting key for the front element in the tree, O(1) used to in selection of subproblem."""
        if not self.T.empty():
            return self.T.queue[0].sortingKey
        else:
            return None

    def getNode(self):
        """
        Method returning a node at the top of the priority queue (sorted based on the lt method of the branching node
        class.
        :return: A branching node (BnbNode object) stored in the branching node. If the queue is empty, None is returned
        """
        if not self.T.empty():
            return self.T.get()
        else:
            return None

    def addNode(self, node: BnbNode):
        """
        Method adding a branching node to the branching tree
        :param node: An object of the BnbNode class
        """
        self.T.put(node)
