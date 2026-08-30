from __future__ import annotations
from abc import ABC, abstractmethod
import re
import time
from src.asmo.classes.geom import Bound, Rectangle, AsmoPoint, SearchArea
from shapely import Point, MultiPoint
import shapely
import queue as Q
import shapely
import docplex.mp.model_reader
import docplex.mp.model
import docplex.mp
import numpy as np
import json
import os
from src.asmo.utils.shapely_get_corner_points import simplify_by_angle
from src.asmo.utils.mspMethods import ND_pointsSum2_wrapper, N
import src.asmo.classes.pointsets

CPLEX_MIP_GAP = 0.1  # 0.00001
CPLEX_RELATIVE_MIP_GAP = 0.1  # 0.00001
n_threads = os.cpu_count()
CPLEX_MIP_GAP = 1e-40  # 0.00001
CPLEX_RELATIVE_MIP_GAP = CPLEX_MIP_GAP  # 0.00001


class Solution:
    def __init__(
        self, values: np.ndarray, p_name, x: np.ndarray, classification: str = "unknown"
    ) -> None:
        self.val: np.ndarray = values  # the vector containing point
        self.p_name: str = p_name  # name/key for subproblem
        self.x = {
            p_name: x
        }  # a representation of the solution corresponding to the point
        self.geom = AsmoPoint(values)
        # self.x_docplex = {xi: x[i] for i, xi in enumerate(p._variables)}
        self.classification: str = classification  # point classification 'se','sne'

    def __repr__(self):
        return tuple((float(vi) for vi in self.val)).__repr__()

    def __getitem__(self, item):
        return self.val[item]

    def __iter__(self):
        return self.val.__iter__()

    def __hash__(self):
        return tuple(self.val).__hash__()

    def __as_dict__(self):
        return {
            "val": tuple((float(vi) for vi in self.val)),
            "p": self.p_name,
            "x": tuple(self.x),
            "cls": self.classification,
        }

    def __add__(self, other: Solution):
        # return Solution(self.val + other.val, self.p_name, self.x + other.x)
        return Solution(
            values=self.val + other.val,
            p_name=self.p_name,
            x=np.array([]),
            classification=self.classification,
        )

    def __lt__(self, other: Solution):
        """__lt__. return True if self dominates other (componen-wise) minimization sense

        Args:
            other (Point): other
        """
        if all(self.val == other.val):
            return False
        return all(self.val <= other.val)


class SolutionList(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.statistics = {}

    def get_geoms(self):
        return MultiPoint([p.val for p in self])

    def set_statistics(self):
        self.statistics = {"n": len(self), "max": max(self), "min": min(self)}

    @property
    def dim(self):
        if len(self) > 0:
            return len(self[0].val)
        else:
            return None

    def _as_dict(self):
        return {
            "solutions": [p.__as_dict__() for p in self],
            "statistics": self.statistics,
        }

    def __add__(self, other) -> SolutionList:
        # minkowski sum
        if isinstance(other, SolutionList):
            return SolutionList([s1 + s2 for s1 in self for s2 in other])
        else:
            raise ValueError(f"Cannot add {type(other)} to SolutionList")

    def concat(self, other) -> SolutionList:
        """Concatenate two SolutionLists"""
        if isinstance(other, SolutionList):
            return SolutionList(super().__add__(other))
        else:
            raise ValueError(f"Cannot concatenate {type(other)} to SolutionList")

    # def _lex_min_ND_filter():

    def _lex_sort(self) -> SolutionList:
        # lex sort the list of solutions
        return SolutionList(sorted(self, key=lambda x: (x.val[0], x.val[1])))

    def N(self):
        Y_pointlist = self.as_pointlist()
        Y_N_tuple_set = set([tuple(y.val) for y in N(Y_pointlist)])
        Y_N_solutionList = SolutionList()
        for y in self:
            if tuple(y.val) in Y_N_tuple_set:
                Y_N_solutionList.append(y)
        # remove duplicates wrt the point values (if there are multiple solutions with the same point value, only keep one of them)
        Y_N_unique = SolutionList()
        for y in Y_N_solutionList:
            if not any(all(y.val == y2.val) for y2 in Y_N_unique):
                Y_N_unique.append(y)
        return Y_N_solutionList
        Y = self._lex_sort()
        Yn = []
        for y in Y:
            if Yn == [] or not Yn[-1] < y:
                Yn.append(y)
        return SolutionList(Yn)
        # return the set of non-dominated points
        return SolutionList([s for s in self if not any(s1 < s for s1 in self)])

    def ND_sum(self, other: SolutionList) -> SolutionList:
        # return the non-dominated points of the Minkowski sum of self and other
        Y_self = self.as_pointlist()
        Y_other = other.as_pointlist()
        ND_sum_set = set([tuple(y.val) for y in ND_pointsSum2_wrapper(Y_self, Y_other)])
        ND_sum_solutionList = SolutionList()
        for y1 in Y_self.points:
            for y2 in Y_other.points:
                if tuple(y1.val + y2.val) in ND_sum_set:
                    ND_sum_solutionList.append(y1 + y2)
        return ND_sum_solutionList

    def save_json(self, filename: str):
        """save_json. Saves the pointlist in a json format. Uses the as_dict method

        Args:
            filename (str): filename
        """

        json_str = json.dumps(self._as_dict(), separators=(",", ":"))

        # Calculate size (approx) in bytes
        size_mb = len(json_str.encode("utf-8")) / 1_000_000
        assert (
            size_mb < 100
        ), f"Size of json file is {size_mb} MB. This is too large. Consider reducing the size of the data"

        with open(filename, "w") as json_file:
            json_file.write(json_str)

    @staticmethod
    def from_json(filename: str) -> SolutionList:
        """load_json. Load a json file containing a pointlist
        Args:
            filename (str): filename
        """
        with open(filename, "r") as json_file:
            json_dict = json.load(json_file)

        Y = SolutionList()
        # print(f"{json_dict=}")
        for p in json_dict["solutions"]:
            Y.append(Solution(np.array(p["val"]), p["p"], np.array(p["x"]), p["cls"]))
        Y.statistics = json_dict["statistics"]
        # print(f"{Y=}")
        return Y

    def get_induced_upper_bound(self) -> Bound:
        YU = sorted(self, key=lambda x: x.val[0])
        U_points = []
        U_points.append([YU[0].val[0], YU[0].val[1] + 10])  # TODO: Why +10
        for k in range(len(YU) - 1):
            U_points.append(YU[k].val)
            U_points.append([YU[k + 1].val[0], YU[k].val[1]])
        U_points.append(YU[-1].val)
        U_points.append([YU[-1].val[0] + 10, YU[-1].val[1]])
        return Bound([AsmoPoint(y) for y in U_points])

    def get_nadir_point(self) -> Solution:
        # get the nadir point of the solution set
        return Solution(
            [max([y.val[0] for y in self]), max([y.val[1] for y in self])],
            self[0].p_name,
            np.array([]),
            classification="nadir",
        )

    def get_ideal_point(self) -> Solution:
        # get the ideal point of the solution set
        return Solution(
            [min([y.val[0] for y in self]), min([y.val[1] for y in self])],
            self[0].p_name,
            np.array([]),
            classification="ideal",
        )

    def get_supported(self, types="extreme") -> SolutionList:
        # return the set of EXTREME supported solutions
        if len(self) == 0:
            return self
        conv_hull = shapely.geometry.MultiPoint(
            [y.val for y in self] + [self.get_nadir_point().val]
        ).convex_hull.boundary

        if types == "extreme":
            # calculates the sef of extreme points of the convex hull
            conv_hull = MultiPoint(simplify_by_angle(conv_hull, deg_tol=0))

            return SolutionList(
                # [y for y in self if shapely.contains_xy(conv_hull, x=y.val[0], y=y.val[1])]
                [y for y in self if conv_hull.distance(Point(y.val)) < 0.1]
            )
        else:
            return SolutionList(
                [
                    y
                    for y in self
                    if shapely.contains_xy(conv_hull, x=y.val[0], y=y.val[1])
                ]
                # [y for y in self if conv_hull.distance(Point(y.val)) < 0.1]
            )
        #
        # max_y1 = n.R.lr[0]
        # max_y2 = n.R.ul[1]
        # convex_hull = shapely.geometry.MultiPoint(
        #     [y.val for y in nY]
        #     + [
        #         np.array((max_y1 + offset, max_y2)),
        #         np.array((max_y1, max_y2 + offset)),
        #         np.array((max_y1 + offset, max_y2 + offset)),
        #     ]
        # ).convex_hull.boundary
        # convex_hull = MultiPoint(simplify_by_angle(convex_hull, deg_tol=0.01))
        #
        # # convex_hull = shapely.geometry.MultiPoint(convex_hull.coords)
        # # get points on the border of the convex hull
        # nY = SolutionList([y for y in nY if convex_hull.distance(Point(y.val)) < 0.1])

    @staticmethod
    def from_coords_dummy(coords, sp_name: str) -> SolutionList:
        return SolutionList([Solution(np.array(y), sp_name, "Dummy") for y in coords])

    @staticmethod
    def from_pointlist_dummy(pointlist, sp_name: str) -> SolutionList:
        return SolutionList([Solution(y.val, sp_name, "Dummy") for y in pointlist])

    def region_of_interest(self, W, gamma: float = 0):

        Yse = self.get_supported()

        y_ul = min(self, key=lambda p: p.val[0])
        y_lr = min(self, key=lambda p: p.val[1])

        y_left = max(
            [point for point in Yse if point.val[0] <= W], key=lambda p: p.val[0]
        )
        y_right = min(
            [point for point in Yse if point.val[0] >= W], key=lambda p: p.val[0]
        )
        f = gamma  # fraction
        R = {
            "ul": (1 - f) * y_left.val + f * y_ul.val,
            "lr": (f) * y_lr.val + (1 - f) * y_right.val,
        }
        return Rectangle(ul=R["ul"], lr=R["lr"])

    def as_pointlist(self) -> src.asmo.classes.pointsets.PointList:
        return src.asmo.classes.pointsets.PointList(
            [src.asmo.classes.pointsets.Point(y.val) for y in self]
        )

    # add slice support returning SolutionList
    def __getitem__(self, item):
        if isinstance(item, slice):
            return SolutionList(super().__getitem__(item))
        else:
            return super().__getitem__(item)


class BnbNode:  # delsøgeområde
    """This class implements a simple branch and bound node. This node will store information about the subproblem
    corresponding to the node
    """

    def __init__(self, p: Problem, nR: Rectangle):
        self.p: Problem = p  # problem associated with node (previous)
        # self.hL: Bound = hL  # lower bound region associated witht the node
        self.R: Rectangle = nR  # rectangle associated with the node
        self.tY: SolutionList = SolutionList()  # set of solutions returned by search
        self.tL: Bound = Bound()  # lower bound after search
        self.tU: Bound = Bound()  # upper bound after search
        self.T: BranchingTree = (
            BranchingTree()
        )  # branching tree associated with the node
        self.depth = 0  # The depth of the node in the branching tree. Useful when searching depth first/breadth first
        self.sortingKey = (
            nR.geom.area
        )  # Can be used if nodes should be sorted based on this key
        self.statistics = {"IP-calls": 0, "IP-time": 0}  #

    def initializeNode(self, farther: "node", branchingVar: int, value: int):
        """
        This method initializes the node-object
        :param farther: a node corresponding to the farther node of the node that should be initialized
        :param branchingVar: the variable that is being branched on when creating this node
        :param value: value that the branching variable should be fixed to (must be either 0 (zero) or 1 (one))
        """
        self.depth = farther.depth + 1
        # self.fixedToOne = list(farther.fixedToOne)
        # self.fixedToZero = list(farther.fixedToZero)
        # self.free = [i for i in farther.free if i != branchingVar]
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
        return self.sortingKey > other.sortingKey

    def check_if_empty(self, L: Bound, U: Bound):
        """
        A test to check if the search area is empty. True if the branching node search area intersected with the search area A(L,U) is empty.
        Checks if the branching node is empty by intersecting the search area defined by the branching node with the search area A(L,U).
        For documentation see docs/documentation/problem-BnbNode_check_if_empty.md

        Args:
            L: Lower bound set of subproblen n.L
            U: Upper bound set of subproblem b.U

        returns:
            True if empty, else False
        """

        A = SearchArea.from_bound_sets(self, L, U)
        # return False # For testing

        if __debug__ and True:
            if A.is_empty:
                for y in self.p.Y_mgs:
                    if self.R.geom.covers(Point(y.val)):
                        print(
                            f"Point {y} is covered by the rectangle {self.R}. This should not happen."
                        )
                        # raise ValueError(
                        #     "MGS point is covered by the rectangle. This should not happen."
                        # )
                        #
                    assert not self.R.geom.covers(
                        Point(y.val)
                    ), f"Point {y} is covered by the rectangle {self.R}. This should not happen."
        return A.is_empty


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

    def getNode(self) -> BnbNode | None:
        """
        Method returning a node at the top of the priority queue (sorted based on the lt method of the branching node
        class.
        :return: A branching node (BnbNode object) stored in the branching node. If the queue is empty, None is returned
        """
        if not self.T.empty():
            return self.T.get()
        else:
            return None

    def __getitem__(self, item):
        return self.T.queue[item]

    def __len__(self):
        return len(self.T.queue)

    def addNode(self, node: BnbNode):
        """
        Method adding a branching node to the branching tree
        :param node: An object of the BnbNode class
        """
        self.T.put(node)


class Problem(ABC):  # s problem
    @abstractmethod
    def load_IP(self, name: str):
        pass

    def __init__(self, name: str):
        self.name = name
        self.load_IP(name)  # setup IP model
        # self.node_selection_strategy: str = "largest"
        self.L: Bound = Bound()  # lower bound for subproblem
        self.U: Bound = Bound()  # upper bound for subproblem
        self.T: BranchingTree = BranchingTree()  # search tree for subproblem
        self.hY: SolutionList = SolutionList()  # set of known nondominated solutions
        self.solutions_preloaded = (
            False  # Flag to indicate if solutions are preloaded (for testing)
        )
        self.Yse: SolutionList = SolutionList()  # set of supported nondominated points
        self.Yn: SolutionList = SolutionList()  # set of globally nondominated points
        self.max_Z: int = (
            10**10
        )  # upper bound for the box [max_Z, max_Z] which contain all of p.Yn
        self.y_ul: Solution | None = None
        self.y_lr: Solution | None = None
        self.search_method: str = "phase1"
        self.statistics = {
            "IP-calls": 0,
            "IP-time": 0,
            "IP-infeasible": 0,
            "nodes explored": 0,
        }  #

    def _phase_one(self):
        pass

    @abstractmethod
    def get_supported(self):
        raise NotImplementedError("This method should be implemented in the subclass")
        return SolutionList()

    def select_next_node(self) -> BnbNode | None:
        # uses self. node_selection_strategy
        return self.T.getNode()

    def _set_initial_lower_bound(self):
        """docstring for initial_lower_bound as conv(p.Yse)_N"""
        # assert p.Yse is not None
        # self.L = Bound(self.Yse)
        self.L = Bound([AsmoPoint(y.val) for y in self.Yse])

    def _local_nadir_points(self, points: SolutionList) -> list:
        # assert all(
        #     points[k].val[0] <= points[k + 1].val[0] for k in range(len(points) - 1)
        # ), "Points must be sorted in increasing order of first objective"
        # if len(points) == 0:
        # raise ValueError("Points list is empty")
        # if len(points) == 1:
        # raise ValueError("Points list contains only one point")
        # if len(points) == 2:
        # return [(points[1].val[0], points[0].val[1])]

        # assert p.Yse
        # U = Bound([(p.Yse[k],p.Yse[k+1]) for k in range(len(p.Yse)-1)]) # # TODO: Lower part of convex hull
        # test the list is sortet decreasing in x
        points = points._lex_sort()
        U_points = []
        for k in range(len(points) - 1):
            U_points.append(points[k])
            U_points.append([points[k + 1].val[0], points[k].val[1]])
        U_points.append(points[-1])

        U = [y for y in U_points]
        return U

    def _set_initial_upper_bound(self):
        """docstring for initial_upper_bound as Yse"""
        # self.U = Bound(self._local_nadir_points(self.Yse))
        self.U = self.Yse.get_induced_upper_bound()

    def partition_search_area(
        self, points: SolutionList
    ):  # modify p.T using p.L and p.hY
        """Partition the search area into subregions"""
        # self.T = BranchingTree()
        points = SolutionList(
            sorted(points, key=lambda x: x.val[0])
        )  # lex sort the input set
        for k in range(len(points) - 1):
            # print("Creating initial branch")
            r_ul = points[k].val + np.array([1, -1])
            r_lr = points[k + 1].val + np.array([-1, 1])
            if not (
                r_ul[0] <= r_lr[0] and r_ul[1] >= r_lr[1]
            ):  # skip if invalid rectangle
                continue
            nR = Rectangle(AsmoPoint(r_ul), AsmoPoint(r_lr))
            # hL = nR.intersection(self.L)
            n = BnbNode(self, nR)

            if False:  # removed 28. sep
                if n.check_if_empty(self.L, self.U):
                    print("Empty node: Skipping")
                    continue
            self.T.addNode(n)
            # print(f"Added initial branching node: now {len(self.T)} nodes in tree")

    # def partition_search_area(self, points: SolutionList):
    # modify p.T using p.L and p.hY
    # pass

    def _set_root_branching_node(self):
        """Partition the search area into subregions"""
        nR = Rectangle(AsmoPoint((0, self.max_Z)), AsmoPoint((self.max_Z, 0)))
        nR = Rectangle(AsmoPoint((0, 10000)), AsmoPoint((10000, 0)))
        n = BnbNode(self, nR)
        self.T.addNode(n)

    def initialize_subproblem(self) -> None:  # modify p.Yse, p.hY, p.max_Z, p.T
        """initialise: run phase 1 for each subproblem and define bound sets"""
        # NOTE: placed here as it uses external method - might not be fitting for in classes.py due to circular imports
        # Run phase 1 method
        self.get_supported()  # get supported points, defining self.Yse
        assert self.Yse  # save set of known ND points
        self.hY = self.Yse

        # initiate bound sets
        self._set_initial_lower_bound()  # conv(p.Yse)_N
        self._set_initial_upper_bound()  # 'lower boundary' of p.Yse + Rpp

        # partition search space NOTE: no branching here
        # self._set_initial_search_tree()

    def _update_bounds_from_node(self, n: BnbNode):

        # TODO: Consider using the the merge bound sets algorithms?

        # n.tL = n.tL.merge_lower_bounds(Bound(self.L.geom.intersection(n.R.geom)))
        # n.tL = n.tL.merge_lower_bounds(self.L)

        if False:
            left = [p for p in self.L.coords if p[0] <= n.R.ul[0]]
            middle = [p for p in n.tL.coords]
            left_middle_nadir = (
                [
                    AsmoPoint(
                        middle[0][0],
                        left[-1][1],
                    )
                ]
                if left
                else []
            )

            right = [p for p in self.L.coords if p[0] >= n.R.lr[0]]
            middle_right_nadir = (
                [AsmoPoint(right[0][0], middle[-1][1])] if right else []
            )
            self.L = Bound(
                left + left_middle_nadir + middle + middle_right_nadir + right
            )
            # self.U.coords = [u for u in self.U.coords if not any( all((um[0] < u[0], um[1] < u[1])) for um in self.U.coords)] # TODO: Not nessesary (should be weakly stable)
        else:
            assert isinstance(n.tL, Bound)
            assert isinstance(self.L, Bound)
            self.L = self.L.merge_lower_bounds(n.tL)
        if False:
            self.U.coords = [
                u
                for u in self.U.coords
                if not any(all((um[0] < u[0], um[1] < u[1])) for um in n.tU.coords)
            ]
            left = [p for p in self.U.coords if p[0] <= n.R.ul[0]]
            middle = [p for p in n.tU.coords]
            left_middle_ideal = (
                [
                    AsmoPoint(
                        left[-1][0],
                        middle[0][1],
                    )
                ]
                if left
                else []
            )
            right = [p for p in self.U.coords if p[0] >= n.R.lr[0]]
            middle_right_ideal = (
                [AsmoPoint(middle[-1][0], right[0][1])] if right else []
            )
            self.U = Bound(
                left + left_middle_ideal + middle + middle_right_ideal + right
            )
        else:
            self.U = self.U.merge_upper_bounds(n.tU)

        # update known solutions
        for y in n.tY:
            if y not in self.hY:
                self.hY.append(y)
        # self.hY = SolutionList(list(self.hY) + list(n.tY))

    def solve_and_reduce(self) -> Solution | None:
        """
        Solve single-objective lambda/epsilon problem
        :param n: a problem to be searched and reduced, in the search area defined by n.
        :type BnbNode

        .. note::
        This function modifies the attribute `p.U`, `p.L` and `p.T`.

        """

    def _search_phase_one(self, n: BnbNode):
        """search_phase_one. Search the node n using phase one method

        This methods applies the phase one method for finding all supported points inside the branching node n, which might be unsupported in the subproblem.
        This is done by adding a constraint on each objective $f1(x) <= R1$ and $f2(x) <= R2$ (restricting solutions to be inside the box).
        After a set of solutions is found (potentially none) the lower and upper bound sets are constructed/updated using these solutions.

        For explanation see: docs/documentation/problem-_search_phase_one.md

        Flags:
            self.solutions_preloaded: If True, the method uses preloaded solutions from self.Yn to define the set of solutions inside the branching node n. Otherwise, a TODO: implement actual search using IP solver

        Args:
            n: a search node

        Raises:
            NotImplementedError: [TODO:description]
        """

        # test
        if self.solutions_preloaded:
            # return all y in Yn if y[0] ub of _y1 and y[y] ub of _y2. Also, only the supported points of this set
            # bounds should be from branching_node n
            # define the points inside the branching node
            nY = SolutionList(
                [y for y in self.Yn if y[0] <= n.R.lr[0] and y[1] <= n.R.ul[1]]
            )

            if True:  # old code
                # only get supported points of nY, ie. points for which there exists a posisitve vector (l1,l2) such that y in min(l1*y[0]+l2*y[1] | for y in nY)
                offset = 5
                max_y1 = n.R.lr[0]
                max_y2 = n.R.ul[1]
                convex_hull = shapely.geometry.MultiPoint(
                    [y.val for y in nY]
                    + [
                        np.array((max_y1 + offset, max_y2)),
                        np.array((max_y1, max_y2 + offset)),
                        np.array((max_y1 + offset, max_y2 + offset)),
                    ]
                ).convex_hull.boundary
                convex_hull = MultiPoint(simplify_by_angle(convex_hull, deg_tol=0.01))

                # convex_hull = shapely.geometry.MultiPoint(convex_hull.coords)
                # get points on the border of the convex hull
                nY = SolutionList(
                    [y for y in nY if convex_hull.distance(Point(y.val)) < 0.1]
                )

            if nY:
                # self.statistics["IP-calls"] += len(nY) * 2 - 1
                n.statistics["IP-calls"] += len(nY) * 2 - 1
            else:
                # self.statistics["IP-calls"] += 1
                n.statistics["IP-calls"] += 1
                self.statistics["IP-infeasible"] += 1

        else:
            # not implemented
            # add e constraints to the IP model to restrict the search to the box defined by n.R
            # if not attr(self, "model"):
            if not hasattr(self, "model"):
                self.load_IP(self.name)  # reload the model to reset constraints
            # set constraint
            self.change_eps_rhs(n.R.lr[0], 0)
            self.change_eps_rhs(n.R.ul[1], 1)

            # run phase 1 constrained to box
            self.get_supported_node(n)

            nY = n.tY
            # raise NotImplementedError("Not implemented - only preload implemented")

        # nY = nY.get_supported()
        # debug
        if (
            len(
                SolutionList(
                    [y for y in self.Yn if y[0] <= n.R.lr[0] and y[1] <= n.R.ul[1]]
                )
            )
            > 0
        ):
            # print(n)
            assert len(nY) > 0
        n.tY = nY

        # set lower bound
        if nY:

            nY_ul = min(nY, key=(lambda x: x[0]))
            nY_lr = min(nY, key=(lambda x: x[1]))

            n.tL = Bound(
                [AsmoPoint(n.R.ul.val)]
                + [AsmoPoint(nY_ul[0], n.R.ul[1])]
                + [AsmoPoint(y.val) for y in nY]
                + [AsmoPoint(n.R.lr[0], nY_lr[1])]
                + [AsmoPoint(n.R.lr.val)]
            )

            # set upper bound
            n.tU = Bound(
                [n.R.ul]
                + [AsmoPoint(n.R.ul[0], nY_ul[1])]
                + self._local_nadir_points(nY)
                # + [AsmoPoint(nY_lr[0], nY_ul[1])]
                + [AsmoPoint(nY_lr[0], n.R.lr[1])]
                + [n.R.lr]
            )

        else:
            # set lower bound
            n.tL = Bound(
                [AsmoPoint(n.R.ul.val)]
                + [AsmoPoint(n.R.lr[0], n.R.ul[1])]
                + [AsmoPoint(n.R.lr.val)]
            )
            # set upper bound
            n.tU = Bound(
                [AsmoPoint(n.R.ul.val)]
                + [AsmoPoint(n.R.ul[0], n.R.lr[1])]
                + [AsmoPoint(n.R.lr.val)]
            )

    def _search_bbm(self, n: BnbNode):
        """search the node using the balanced-box mehtod (bbm)
        Args:
            n (BnbNode): node to be searched
        Modifies:
            n.tY (SolutionList): set of solutions returned by search
            n.tL (Bound): lower bound after search
            n.tU (Bound): upper bound after search
            n.T (BranchingTree): branching tree associated with the node
        """
        # c0 = 0.5*(n.R.ul[0] + n.R.lr[0]) # center on first coord
        c1 = 0.5 * (n.R.ul[1] + n.R.lr[1])  # center on second coord

        Rb = Rectangle(AsmoPoint((n.R.ul[0], c1)), n.R.lr)

        # Weight on 'other' objective to ensure lex-mininimization
        alpha1 = 1 / (n.R.lr[0] + n.R.ul[0] + 1)  # DEBUG: Should this '-' be '+'?
        alpha2 = 1 / (n.R.ul[1] + n.R.lr[1] + 1)  # DEBUG: Should this '-' be '+'?

        if self.solutions_preloaded:

            y1 = max(
                [y for y in self.Yn if Rb.geom.covers(Point(y.val))],
                key=(lambda x: alpha1 * x[0] + x[1]),
                default=None,
            )
            # self.statistics["IP-calls"] += 1
            n.statistics["IP-calls"] += 1

            n.tY = SolutionList()
            if y1 and (y1.geom.val == n.R.lr.val).all():
                if y1 not in n.tY:
                    n.tY.append(y1)
                y1 = None

            if y1 is not None:
                Rt = Rectangle(n.R.ul, AsmoPoint((y1[0] - 1, c1)))
            else:
                Rt = Rectangle(n.R.ul, AsmoPoint((Rb.lr[0], c1)))
            y2 = max(
                [y for y in self.Yn if Rt.geom.covers(Point(y.val))],
                key=(lambda x: x[0] + alpha2 * x[1]),
                default=None,
            )
            # self.statistics["IP-calls"] += 1
            n.statistics["IP-calls"] += 1

            if y2 and (y2.geom.val == n.R.ul.val).all():
                if y2 not in n.tY:
                    n.tY.append(y2)
                y2 = None

            # n.statistics["IP-calls"]

        else:
            # not implemented
            # raise NotImplementedError("Not implemented - only preload implemented")
            self.change_eps_rhs(n.R.lr[0], 0)  # add constraint to IP model
            self.change_eps_rhs(c1, 1)  # add constraint to IP model
            self.set_single_objective(
                1 / alpha1
            )  # set objective to balanced box method objective
            if self.solve_single_objective():
                y1 = self.retrieve_solution()
            else:
                y1 = None

            self.set_single_objective(
                alpha2
            )  # set objective to balanced box method objective
            self.change_eps_rhs(n.R.ul[1], 1)  # add constraint to IP model
            if y1 is None:
                self.change_eps_rhs(Rb.lr[0], 0)
            else:
                self.change_eps_rhs(y1[0] - 1, 0)

            if self.solve_single_objective():
                y2 = self.retrieve_solution()
            else:
                y2 = None
            # x1 = self.solve_single_objective()  # solve IP model with new constraint

        if y1 is not None:
            n.tY.append(y1)
        if y2 is not None:
            n.tY.append(y2)

        if True:
            # add bound sets
            if not (y1 in n.tY or y2 in n.tY):
                # no points found – fallback box bounds
                n.tL = Bound([n.R.ul, AsmoPoint((n.R.lr[0], n.R.ul[1])), n.R.lr])
                n.tU = Bound([n.R.ul, AsmoPoint((n.R.ul[0], n.R.lr[1])), n.R.lr])

            elif y1 and y2:
                # simple polygon through UL -> y2 -> y1 -> LR
                n.tL = Bound(
                    [
                        n.R.ul,
                        AsmoPoint((n.R.ul[0], y2[1])),
                        AsmoPoint(y2.val),
                        AsmoPoint((y1[0], y2[1])),
                        AsmoPoint(y1.val),
                        AsmoPoint((y1[0], n.R.lr[1])),
                        n.R.lr,
                    ]
                )
                n.tU = Bound(
                    [
                        n.R.ul,
                        AsmoPoint((y2[0], n.R.ul[1])),
                        AsmoPoint(y2.val),
                        AsmoPoint((y2[0], y1[1])),
                        AsmoPoint(y1.val),
                        AsmoPoint((n.R.lr[0], y1[1])),
                        n.R.lr,
                    ]
                )

            elif y1:
                # exactly one point p
                p = y1
                n.tL = Bound(
                    [
                        n.R.ul,
                        AsmoPoint((p[0], n.R.ul[1])),
                        AsmoPoint(p.val),
                        AsmoPoint((p[0], n.R.lr[1])),
                        n.R.lr,
                    ]
                )
                n.tU = Bound(
                    [
                        n.R.ul,
                        AsmoPoint((n.R.ul[0], p[1])),
                        AsmoPoint(p.val),
                        AsmoPoint((n.R.lr[0], p[1])),
                        n.R.lr,
                    ]
                )
            elif y2:
                # exactly one point p
                p = y2
                n.tU = Bound(
                    [
                        n.R.ul,
                        AsmoPoint((p[0], n.R.ul[1])),
                        AsmoPoint(p.val),
                        AsmoPoint((p[0], n.R.lr[1])),
                        n.R.lr,
                    ]
                )
                n.tL = Bound(
                    [
                        n.R.ul,
                        AsmoPoint((n.R.ul[0], p[1])),
                        AsmoPoint(p.val),
                        AsmoPoint((n.R.lr[0], p[1])),
                        n.R.lr,
                    ]
                )
            # create branching nodes
            Branching_rectangles = []
            ul_offset = AsmoPoint((-1, 1))  # offset in direction of ul
            lr_offset = AsmoPoint((1, -1))  # offset in direction of lr
            # ul_offset = AsmoPoint((-0.5, 0.5))  # offset in direction of ul
            # lr_offset = AsmoPoint((0.5, -0.5))  # offset in direction of lr
            # lr_offset = AsmoPoint((0,0)) # offset in direction of lr
            # ul_offset = AsmoPoint((0,0)) # offset in direction of lr

            if False:
                ul_offset = AsmoPoint((0, 0))  # offset in direction of ul
                lr_offset = AsmoPoint((0, 0))  # offset in direction of lr

            if not (y1 or y2):
                pass  # no branching
            elif y1 and y2:
                Branching_rectangles.append(Rectangle(n.R.ul, y2 + ul_offset))
                Branching_rectangles.append(Rectangle(y1 + lr_offset, n.R.lr))
            elif y1:
                Branching_rectangles.append(Rectangle(y1 + lr_offset, n.R.lr))
            elif y2:
                Branching_rectangles.append(Rectangle(n.R.ul, y2 + ul_offset))

            for r in Branching_rectangles:
                if r.geom.is_empty:
                    print("skipping empty rectangle")
                    continue
                if n.tY.get_geoms().covers(r.geom):
                    print("skipping singleton")
                    continue  # skip if the rectangle is covered by the known solutions
                n.T.addNode(BnbNode(n.p, r))

            # merge with current lower bound TODO: merge lower bound to ensure new lb dominates old lb.
            # n.tL = n.tL.merge_lower_bounds(self.L)

        # n.T.addNode(BnbNode(n.p, Rectangle(n.R.ul, y2)))

        if __debug__:  # check that all points in Yn are integer vectors
            for y in self.Yn:
                for yi in y.val:
                    if yi % 1 != 0:
                        raise ValueError(f"Point {y} in Yn is not an integer vector")

    def search_node(self, n: BnbNode):
        """search_node. Search the node n using chosen search strategy
        Args:
            n (BnbNode): node to be searched
        """

        match self.search_method:
            case "phase1":
                self._search_phase_one(n)
            case "bbm":
                self._search_bbm(n)
            case "random":
                choices = [self._search_phase_one, self._search_bbm]
                method = np.random.choice(choices)
                method(n)
            case "epsilon1":
                raise ValueError(f"not implemented method {self.search_method}")
            case "epsilon2":
                raise ValueError(f"not implemented method {self.search_method}")
            case "bi-dir-epsilon":
                raise ValueError(f"not implemented method {self.search_method}")
            case _:
                raise ValueError(f"Unknown search method {self.search_method}")


class CVRP(Problem):
    instance_dir = "./../instances/cvrp/IP/"
    instance_dir = "./instances/cvrp/IP/"
    Yse_solution_dir = "./instances/cvrp/Yse/"
    Yn_solution_dir = "./instances/cvrp/Yn/"

    M: int = 1000000  # large M

    def verbose(self, val: bool):
        self.ip_model.context.solver.log_output = val

    def load_IP(self, name: str):
        # remove _1 or _2 ... from end of name (if solving the same instance multiple times for testing) . Only of ending with _[0-9]
        stripped_name = re.sub(r"_[0-9]+$", "", name)
        self.ip_model: docplex.mp.model.Model = (
            docplex.mp.model_reader.ModelReader.read(
                os.path.join(self.instance_dir, stripped_name)
            )
        )
        self._f1_constr: docplex.mp.model.LinearConstraint = (
            self.ip_model.get_constraint_by_name("f1")
        )
        self._f2_constr: docplex.mp.model.LinearConstraint = (
            self.ip_model.get_constraint_by_name("f2")
        )
        self._y1 = self.ip_model.get_var_by_name("y1")
        self._y2 = self.ip_model.get_var_by_name("y2")
        self._variables = tuple(self.ip_model.iter_variables())
        self._coefficients = np.array(
            [
                [self._f1_constr.lhs.get_coef(xi) for xi in self._variables],
                [self._f2_constr.lhs.get_coef(xi) for xi in self._variables],
            ]
        )
        self._coefficients.flags.writeable = False  # make immutible we do not want to change the initial weights, only the objective and epsilon constraints

    def change_eps_rhs(self, new_rhs: int | float, obj: int = 0):
        # change the rhs of the epsilon constraint for the chosen objective
        match obj:
            case 0:
                self._y1.ub = new_rhs
            case 1:
                self._y2.ub = new_rhs
            case _:
                raise ValueError("obj must be 0 or 1")

    def solve_single_objective(self, verbose=False, timelimit: int | bool = False):
        # solve to optimality with relative tolerance 1e-4
        self.ip_model.parameters.mip.tolerances.mipgap = CPLEX_RELATIVE_MIP_GAP
        self.ip_model.parameters.mip.tolerances.absmipgap = CPLEX_MIP_GAP
        if False:  # for testing
            self.ip_model.parameters.mip.tolerances.mipgap = 0.9
            self.ip_model.parameters.mip.tolerances.absmipgap = 100

        self.ip_model.parameters.threads = max(1, n_threads - 1)
        if timelimit:
            self.ip_model.solve(log_output=verbose, time_limit=timelimit)
        else:
            self.ip_model.solve(log_output=verbose)
        self.statistics["IP-time"] += self.ip_model.solve_details.time
        self.statistics["IP-calls"] += 1
        if self.ip_model.solution is None:
            print(f" ** Problem infeasible ** ")
            self.statistics["IP-infeasible"] += 1
            return False
        else:
            return True

    def eval_solution(self) -> np.ndarray:
        # return np.array(
        # [self.ip_model.solution.get_value(xi) for xi in (self._y1, self._y2)]
        # )
        # round to 6th decimal place
        return np.array(
            [
                round(self.ip_model.solution.get_value(xi), 6)
                for xi in (self._y1, self._y2)
            ]
        )

    def set_single_objective(self, l):
        m = self.ip_model
        m.minimize(l * self._y1 + self._y2)

    def set_lexmin(self, obj_index: int):
        if self.solutions_preloaded:
            print(f"Solutions preloaded, skipping lexmin for objective {obj_index}")
            try:
                self.y_lr = [y for y in self.Yse if y.classification == "y_lr"][0]
                self.y_ul = [y for y in self.Yse if y.classification == "y_ul"][0]
            except IndexError:
                self.y_lr = min(self.Yse, key=lambda y: y[1])
                self.y_ul = min(self.Yse, key=lambda y: y[0])

            return

        if obj_index == 0:  # y_lr
            self.set_single_objective(1 / self.M)
            if self.solve_single_objective():
                self.y_lr = self.retrieve_solution()
                self.y_lr.classification = "y_lr"

        elif obj_index == 1:  # y_ul
            self.set_single_objective(self.M)
            if self.solve_single_objective():

                self.y_ul = self.retrieve_solution()
                self.y_ul.classification = "y_ul"

    def retrieve_solution(self):
        decimal_precision = 4
        return Solution(
            self.eval_solution(),
            self.name,
            np.array(
                [
                    np.round(self.ip_model.solution.get_value(xi), decimal_precision)
                    for xi in self._variables
                ]
            ),
        )

    def _load_supported_solution(self) -> bool:
        # load Yse from solution dir and set self.Yse, self.hY, self.y_ul, self.y_lr
        if not os.path.exists(f"{self.Yse_solution_dir}{self.name}.json"):
            return False
        self.Yse = SolutionList.from_json(f"{self.Yse_solution_dir}{self.name}.json")
        self.hY = self.Yse
        # set self.y_ul to the element of self.Yse wich classification == 'y_ul'
        self.y_ul = next((y for y in self.Yse if y.classification == "y_ul"), None)
        self.y_lr = next((y for y in self.Yse if y.classification == "y_lr"), None)
        return True

    def get_supported_node(self, n: BnbNode):
        """get_supported. Get the supported points of the problem"""

        # y_ul = Solution(n.R.ul.val, self.name, classification="y_ul_n", x="dummy")
        # y_lr = Solution(n.R.lr.val, self.name, classification="y_lr_n", x="dummy")

        # get y_lr
        self.set_single_objective(1 / self.M)
        if self.solve_single_objective():
            y_lr = self.retrieve_solution()
        else:
            n.tY = SolutionList()
            # self.statistics["IP-calls"] += 1
            n.statistics["IP-calls"] += 1
            # self.statistics["IP-infeasible"] += 1
            return
        # get y_ul
        self.set_single_objective(self.M)
        if self.solve_single_objective():
            # self.statistics["IP-calls"] += 1
            n.statistics["IP-calls"] += 1
            y_ul = self.retrieve_solution()

        if (y_ul.geom.val == y_lr.geom.val).all():
            n.tY = SolutionList([y_ul])
            return

        y_left = y_ul
        y_right = y_lr
        Yse = [y_left, y_right]  # initiate linked list
        # main loop
        while y_left.geom != y_lr.geom:
            if __debug__:
                print(
                    f"y_left: {y_left}, y_right: {y_right}, searching between {y_left} and {y_right}"
                )
                print(f"{y_lr=}, {y_ul=}")
                print(f"{y_lr==y_ul=}")
            l = (y_left[1] - y_right[1]) / (y_right[0] - y_left[0])

            print(f"{l=}")

            # if l is na
            if np.isnan(l) or np.isinf(l):
                if __debug__:
                    print(f"l is {l}, skipping search between {y_left} and {y_right}")
                raise
            self.set_single_objective(l)

            self.solve_single_objective()
            # self.statistics["IP-calls"] += 1
            n.statistics["IP-calls"] += 1
            y_star = self.retrieve_solution()
            y_star.classification = "n"

            if (
                l * y_star[0] + y_star[1] < l * y_left[0] + y_left[1] - 0.001
            ):  # take cplex precision error into account
                # add to Yse between y_left and y_right
                if __debug__:
                    print(f"new point {y_star} found between {y_left} and {y_right}")
                Yse.insert(Yse.index(y_left) + 1, y_star)
                y_right = y_star
            else:
                y_left = y_right
            y_right = Yse[min(Yse.index(y_left) + 1, len(Yse) - 1)]

        n.tY = SolutionList(Yse)
        # self.Yse = SolutionList(Yse)
        # self.hY = SolutionList(Yse)

        # self.Yse.save_json(f"{self.Yse_solution_dir}{self.name}.json")

    def get_supported(self, timelimit=False):
        """get_supported. Get the supported points of the problem"""

        print(f"{self.solutions_preloaded=}")
        if self.solutions_preloaded:
            return self.Yse
        if self.y_ul is None:
            self.set_lexmin(1)
        if self.y_lr is None:
            self.set_lexmin(0)

        Yse = [self.y_ul, self.y_lr]  # initiate linked list
        print(f"{Yse=}")
        y_left, y_right = Yse[0], Yse[1]

        assert self.y_ul is not None and self.y_lr is not None, "y_ul or y_lr not set"
        # main loop

        max_time = time.time() + timelimit if timelimit else None
        while y_left != self.y_lr:

            l = (y_left[1] - y_right[1]) / (y_right[0] - y_left[0])

            print(f"{l=}")

            self.set_single_objective(l)
            if timelimit:
                self.solve_single_objective(timelimit)
            else:
                self.solve_single_objective()
            if timelimit and time.time() > max_time:
                print("Time limit reached, stopping get_supported")
                return False
            y_star = self.retrieve_solution()
            y_star.classification = "se"

            if (
                l * y_star[0] + y_star[1] < l * y_left[0] + y_left[1] - 0.001
            ):  # take cplex precision error into account
                # add to Yse between y_left and y_right
                if __debug__:
                    print(f"new point {y_star} found between {y_left} and {y_right}")
                Yse.insert(Yse.index(y_left) + 1, y_star)
                y_right = y_star
            else:
                y_left = y_right
            y_right = Yse[min(Yse.index(y_left) + 1, len(Yse) - 1)]

        self.Yse = SolutionList(Yse)
        self.hY = SolutionList(Yse)

        self.Yse.save_json(f"{self.Yse_solution_dir}{self.name}.json")
        return True

    def epsilon_constraint(self, delta=1, timelimit=False):
        """epsilon_constraint.

        Args:
            delta: smallest difference of obj2
        """

        if __debug__:
            print(f"Running epsilon constraint method: {delta=}")

        # self.initiate_eps_constr(obj = '1') # add constraint
        # self.set_single_objective(1)  # reset objective weights
        self.set_single_objective(self.M)

        Yn = []
        # eps method
        time_before = time.time()
        max_time = time_before + timelimit if timelimit else None
        while self.solve_single_objective(
            timelimit=(max_time - time.time() if max_time else False)
        ):
            if timelimit and time.time() > max_time:
                print("Time limit reached, stopping epsilon constraint method")
                return False
            y = self.retrieve_solution()
            Yn.append(y)
            print(f"New solution found {y=}")
            self.change_eps_rhs(y.val[1] - delta, obj=1)
            if __debug__:
                print(f"   New solution {y=}, {len(Yn)=}")
        self.hY = SolutionList(Yn)

        return True

    def preload_solutions(self):
        # load Yn from solution dir and set self.Yn
        if os.path.exists(f"{self.Yn_solution_dir}{self.name}_preloaded.json"):
            self.Yn = SolutionList.from_json(
                f"{self.Yn_solution_dir}{self.name}_preloaded.json"
            )
        elif os.path.exists(f"{self.Yn_solution_dir}{self.name}.json"):
            self.Yn = SolutionList.from_json(f"{self.Yn_solution_dir}{self.name}.json")
        else:
            raise FileNotFoundError(f"{self.Yn_solution_dir}{self.name}.json not found")
            # self.Yn = SolutionList()
        # prload Yse
        if os.path.exists(f"{self.Yse_solution_dir}{self.name}.json"):
            self.Yse = SolutionList.from_json(
                f"{self.Yse_solution_dir}{self.name}.json"
            )
        else:
            # raise FileNotFoundError(
            #     f"{self.Yse_solution_dir}{self.name}.json not found"
            # )
            self.Yse = SolutionList()
            if self.Yn:
                self.Yse = self.Yn.get_supported()

        # check that
        self.solutions_preloaded = True  # set flag
