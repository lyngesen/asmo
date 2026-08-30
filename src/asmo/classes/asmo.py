from src.asmo.utils import mspMethods
from src.asmo.utils.timing import print_timeit
from src.asmo.classes.geom import Bound, SearchArea, Rectangle
from src.asmo.classes.problem import Problem, Solution, SolutionList, AsmoPoint, CVRP
from src.asmo.classes.pointsets import MinkowskiSumProblem, PointList

# from src.asmo.utils.upper_envelope import upper_envelope
from src.asmo.utils.mspMethods import induced_UB, ND_pointsSum2_wrapper
import json
import time
import shapely
from shapely import Polygon
from functools import reduce
from src.asmo.classes.plotter import Plotter
import random
from src.asmo.utils.fastMinimumGenerator import algorithm2, Y_list_to_fixed_reduced
from src.asmo.utils.fast_A_dominated_by_B import check_dominated_A_by_B


def add_upper(a, b):
    import math

    self = a
    other = b
    coords = [
        self[0].val + other[0].val,
        self[0].val + other[1].val,
        self[1].val + other[1].val,
        self[1].val + other[0].val,
    ]

    # Compute centroid
    cx = sum(p[0] for p in coords) / len(coords)
    cy = sum(p[1] for p in coords) / len(coords)

    # Sort counter-clockwise around centroid
    coords = sorted(coords, key=lambda p: -math.atan2(p[1] - cy, p[0] - cx))

    # Robustly identify the two *opposite* corners to extend:
    #   upper-left-ish  -> max of (y - x)
    #   lower-right-ish -> min of (y - x)
    def diag_key(p):
        return p[1] - p[0]

    ul = max(coords, key=diag_key)  # upper-left
    lr = min(coords, key=diag_key)  # lower-right

    # Extend toward infinity (same idea as your -100, but direction-aware).
    INF = 1e6  # or 100, to match your existing convention

    new_coords = []
    for p in coords:
        if (p == ul).add():
            # extend "upper-left" to the top: y -> +INF, keep x
            new_coords.append((p[0], INF))
            # If you literally meant "to 0", use:
            # new_coords.append((0, p[1]))   # clamp x to 0
            # or
            # new_coords.append((p[0], 0))   # clamp y to 0
        elif p == lr:
            # extend "lower-right" down: y -> -INF, keep x
            new_coords.append((p[0], -INF))
            # If you literally meant "to 0":
            # new_coords.append((0, p[1]))   # clamp x to 0
        else:
            new_coords.append(p)

    # Re-sort so the polygon stays CCW after the edits
    cx = sum(p[0] for p in new_coords) / len(new_coords)
    cy = sum(p[1] for p in new_coords) / len(new_coords)
    new_coords = sorted(new_coords, key=lambda p: -math.atan2(p[1] - cy, p[0] - cx))

    return Polygon(new_coords)


class ASMO:
    def __init__(self):
        self.P: list[Problem] = []
        self.U: Bound = Bound()
        self.Yn: SolutionList = SolutionList()
        self.Z: int = 10**10
        self.strategy_subproblem_selection: str = "random"  # 'alternating', 'smallest'
        self.strategy_node_priority: str = "largest"  # 'mostPotential', 'smallest'
        self.strategy_reduction: str | int = (
            "all"  # 'previous', int=> update iteration mod int
        )
        self.strategy_update_node_priority: str | int = (
            "all"  # same strategy as above, otherwise above doesnt make too much sense?
        )
        self.strategy_refine_with_integer_gap: bool = False
        self.strategy_search_method = "phase1"
        self.strategy_refine_method = "fast"

        self.ROI: Rectangle | None = None  # Region of Interest
        self.statistics = {"IP-calls": 0, "nodes explored": 0}

    def set_strategy_config(
        self,
        strategy_subproblem_selection="largest",
        strategy_node_priority="largest",
        strategy_reduction="all",
        strategy_update_node_priority="all",
        strategy_refine_with_integer_gap=False,
        strategy_search_method="phase1",
        strategy_refine_method="fast",
    ):
        strategy_subproblem_selection_valid = (
            "largest",
            "alternating",
            "smallest",
            "sequential",
        )
        strategy_node_priority_valid = ("largest", "smallest", "mostPotential")
        strategy_reduction_valid = set(("none", "all", "first")).union(
            range(1000)
        )  # Example valid values
        strategy_update_node_priority_valid = set(("all",)).union(
            range(1000)
        )  # Example valid values

        strategy_search_method_valid = ("phase1", "bbm")
        assert (
            strategy_subproblem_selection in strategy_subproblem_selection_valid
        ), f"{strategy_subproblem_selection} not a valid strategy, must be in {strategy_subproblem_selection_valid}"
        assert (
            strategy_node_priority in strategy_node_priority_valid
        ), f"{strategy_node_priority} not a valid strategy, must be in {strategy_node_priority_valid}"
        assert (
            strategy_reduction in strategy_reduction_valid
        ), f"{strategy_reduction} not a valid strategy, must be in {strategy_reduction_valid}"
        assert (
            strategy_update_node_priority in strategy_update_node_priority_valid
        ), f"{strategy_update_node_priority} not a valid strategy, must be in {strategy_update_node_priority_valid}"
        assert (
            strategy_search_method in strategy_search_method_valid
        ), f"{strategy_search_method} not a valid strategy, must be in {strategy_search_method_valid}"
        assert strategy_refine_with_integer_gap in (
            True,
            False,
        ), f"{strategy_refine_with_integer_gap} not a valid strategy, must be in (True, False)"
        self.strategy_refine_with_integer_gap = strategy_refine_with_integer_gap
        self.strategy_subproblem_selection = strategy_subproblem_selection
        self.strategy_node_priority = strategy_node_priority
        self.strategy_reduction = strategy_reduction
        self.strategy_search_method = strategy_search_method
        self.strategy_update_node_priority = strategy_update_node_priority
        for p in self.P:
            p.search_method = strategy_search_method

        assert strategy_refine_method in (
            "fast",
            "smart",
            "slow",
        ), f"{strategy_refine_method} not a valid strategy, must be in ('fast', 'smart', 'slow')"
        self.strategy_refine_method = strategy_refine_method

    def get_strategy_dict(self):
        return {
            "strategy_subproblem_selection": self.strategy_subproblem_selection,
            "strategy_node_priority": self.strategy_node_priority,
            "strategy_reduction": self.strategy_reduction,
            "strategy_update_node_priority": self.strategy_update_node_priority,
            "strategy_refine_with_integer_gap": self.strategy_refine_with_integer_gap,
            "strategy_search_method": self.strategy_search_method,
            "strategy_refine_method": self.strategy_refine_method,
        }

    def print_strategy_config(self):
        print("ASMO strategy configuration:")
        for key, value in self.get_strategy_dict().items():
            print(f" {key}: {value}")

    def load_subproblem(self, p: Problem):
        self.P.append(p)
        self.U = Bound()  # reset - is no longer a valid UB
        p.search_method = self.strategy_search_method

    def preload_subproblem(self, s, sp_filename):
        subproblem_path = "./instances/subproblems_local_sets/"
        sp_name = sp_filename.replace("subproblems/", "")
        Y = PointList.from_json(subproblem_path + sp_name)
        p = CVRP("CVRP_neg_0_16_8_35_NEW.lp")
        p.preload_solutions()
        p.name = sp_name
        X = SolutionList.from_pointlist_dummy(Y, s)
        p.Yn = X
        p.Yse = X.get_supported()
        return p

    @staticmethod
    def from_json(filename: str):
        asmo = ASMO()
        content = json.load(open(filename, "r"))[0]

        print(content)
        for s, sp_filename in content.items():
            p = asmo.preload_subproblem(s, sp_filename)
            asmo.load_subproblem(p)

        # asmo = ASMO.from_msp_file(filename)
        asmo.statistics["filename"] = filename

        return asmo

    def set_Z_max(self):
        # define axis-max such that [Z,Z]^2 contains all subproblems
        self.Z = max([p.max_Z for p in self.P])

    def _set_incumbent(self) -> None:
        # sets self.Yh
        self.hY = reduce(lambda hY1, hY2: (hY1.ND_sum(hY2)), [p.hY for p in self.P])
        # self.hY = reduce(lambda hY1, hY2: (hY1 + hY2).N(), [p.hY for p in self.P])

    def _set_global_upper_bound(self) -> None:
        # uses p.U for p in self.P
        # local_bounds = [p.U for p in self.P]
        # self.hY = reduce(lambda hY1,hY2:  (hY1+hY2).N() , [p.hY for p in self.P])
        self._set_incumbent()  # sets self.hY
        # self.U = reduce(lambda U1,U2:  SearchArea(U1+U2).get_lower_bound() , local_bounds)
        self.U = self.hY.get_induced_upper_bound()
        self.U.is_stairs = False
        # self.U = Bound()  # TODO
        if self.ROI:
            self.U = self.U.merge_upper_bounds(
                self.ROI.get_upper_bound().extend_upper()
            )

    def _set_global_lower_bound(self) -> None:
        # set global lower bound
        self.L = reduce(
            lambda L1, L2: SearchArea(
                L1.extend_lower() + L2.extend_lower()
            ).get_lower_bound(),
            [p.L for p in self.P],
        )

        if self.ROI:
            self.L = self.L.merge_lower_bounds(
                self.ROI.get_lower_bound().extend_lower()
            )

    def _update_global_lower_bound(self) -> None:
        self._set_global_lower_bound()  # default to recalculation of L

    # modifies U using the new globally nondominated points generated using y
    def _update_global_upper_bound(self, y: Solution) -> None:
        # This should be faster than running set_global_upper_bound TODO: use y
        self._set_global_upper_bound()  # default to recalculation of U

    # modify p.U
    @staticmethod
    def _lower_bound_sum_lower_part_pairs(A: Bound, B: Bound):
        res_ideal = A.get_ideal_points() + B.get_ideal_points()
        res = SearchArea(A + B).get_lower_bound()

    @staticmethod
    def _ms_sum_upper_part_pairs(A: Bound, B: Bound, plot=False):
        # should return the pairs of A and B which could be part in the upper envelope of A+B.
        # A pair of points a+b can not be part of the upper part if their ideal point (max) is strictly dominated by any point of Nadir(A)+Nadir(B) UNION A.coords.as_pointlist + B.coords.as_pointlist.
        # logic. If a+b dominated by a'+b', then a+b is (strictly) under the upper_envelope and therefor not needed to 'generate' it
        # steps:
        #   create list of nadir points (a lower bound)
        #   create list {(a,b) | a+b not dominated by BOUND}
        #   return the set of pairs.
        A_lines = A.get_lines()
        B_lines = B.get_lines()
        A_ideal = PointList([-a.ideal for a in A_lines])
        B_ideal = PointList([-b.ideal for b in B_lines])
        A_nadir = PointList([a.nadir for a in A_lines])
        B_nadir = PointList([b.nadir for b in B_lines])

        # _bound_nadir = PointList((ND_pointsSum2_wrapper(A_nadir, B_nadir)) * (-1))
        if __debug__:
            print("\tbefore bound nadir")
        _bound_nadir = A_nadir + B_nadir
        if __debug__:
            print("\tbefore bound")

        if False:
            _bound = PointList((ND_pointsSum2_wrapper(A_ideal, B_ideal) * (-1)))
        else:
            _sorted_sum = mspMethods.lex_sort(A_ideal + B_ideal)
            _bound = []
            for y in _sorted_sum:
                if _bound == [] or not all((_bound[-1][_i] < y[_i] for _i in range(2))):
                    _bound.append(y)

            _bound = PointList(_bound)

        if __debug__:
            print("\tbefore check dominated")
        a_b_pairs = []

        nd_nadir = set(
            map(
                tuple,
                check_dominated_A_by_B(
                    _bound_nadir.as_np_array(), _bound.as_np_array(), strict=False
                ),
            )
        )

        if __debug__:
            print("\tbefore get nd pairs")
        for a in A_lines:
            for b in B_lines:
                minxa, minya, maxxa, maxya = a.geom.bounds
                minxb, minyb, maxxb, maxyb = b.geom.bounds
                # geom = a + b
                # minx, miny, maxx, maxy = geom.bounds
                # if tuple(-(a.nadir.val + b.nadir.val)) in nd_nadir:
                if tuple((maxxa + maxxb, maxya + maxyb)) in nd_nadir:
                    # if _bound.dominates_point(AsmoPoint((maxxa + minxb, maxya + maxyb))):
                    a_b_pairs.append((a, b))

        if __debug__:
            print("\tafter get nd pairs")
        if plot:
            a_b_pairs_dominated = []
            for a in A.get_lines():
                for b in B.get_lines():
                    geom = a + b
                    minx, miny, maxx, maxy = geom.bounds
                    if not _bound.dominates_point(AsmoPoint((maxx, maxy))):
                        a_b_pairs_dominated.append((a, b))

            P = Plotter()
            for _, (a, b) in enumerate(a_b_pairs):
                geom = a + b
                P.plot(
                    geom,
                    name=f"a+b not dominated {len(a_b_pairs)}" if _ == 0 else "_",
                    color="green",
                )
            for _, (a, b) in enumerate(a_b_pairs_dominated):
                geom = a + b
                P.plot(
                    geom,
                    name=f"a+b dominated {len(a_b_pairs_dominated)}" if _ == 0 else "_",
                    color="red",
                )
            P.plot(A, name="A")
            A_ideal.plot(ax=P.axs[0])
            P.plot(B, name="B")
            B_ideal.plot(ax=P.axs[0])
            P.plot(A + B, name="A+B")
            nd_nadir_points = [AsmoPoint(s) for s in nd_nadir]
            _bound.plot("-N(-ideal(A)-ideal(B))", ax=P.axs[0])
            PointList(nd_nadir_points).plot(ax=P.axs[0], l="nadir", s=0.5)
            P.add_legend()
            P.save("_ms_sum_upper_part_pairs.pdf")
            print(P.filename)

        return a_b_pairs

    def _get_L_hat(self, p: Problem) -> Bound:
        # index of 'other' subproblems not p
        S_hat = [p_other for p_other in self.P if p.name != p_other.name]
        other_lower_bounds = [p.L for p in S_hat]
        # (combined) lower bound of other subproblems
        L_hat = reduce(
            lambda L1, L2: SearchArea(L1 + L2).get_lower_bound(), other_lower_bounds
        )

        return L_hat

    def _reduce_upper_bound(self, p: Problem, version=None) -> None:
        """
        modifies/updates the upper bound p.U using the bound sets of subproblems in P

        See documentation in: docs/documentation/ASMO-_reduce_upper_bound.md

        :param P: all subproblems
        :type Iterable[Problem]
        :param p: subproblem to update
        """

        if version is not None:
            print(f"  Setting reduce_upper_bound method to {version=}")
            self.strategy_refine_method = version

        assert self.strategy_refine_method in ("fast", "smart", "slow")

        L_hat = self._get_L_hat(p)

        # define polygon U - \hat{L}
        if self.strategy_refine_with_integer_gap:
            U = self.U + AsmoPoint((-1, -1))
        else:
            U = self.U

        if self.strategy_refine_method == "slow":
            Ugs = SearchArea(U + (-L_hat))
        elif self.strategy_refine_method == "smart":

            a_b_nd_pairs = ASMO._ms_sum_upper_part_pairs(U, L_hat * -1)
            if True:
                Ugs = SearchArea(
                    shapely.union_all([(a + b) for a, b in a_b_nd_pairs])
                ).get_upper_bound()
            else:

                Ugs = SearchArea(
                    shapely.union_all([add_upper(a, b) for a, b in a_b_nd_pairs])
                ).get_upper_bound()

        if self.strategy_refine_method in ("smart", "slow") and False:
            # remove excess search area. Otherwise a large upper bound set is returned containing points outside the lex min border.
            Ugs = SearchArea(
                # Ugs.geom.intersection(p.U.geom.envelope)
                Ugs.geom.intersection(p.Yse.get_geoms().envelope)
            )  # TODO: Use lex min instead of the geom.envelope

        if self.strategy_refine_method == "slow":
            # get the upper part of the shape - this is the application of the ()_wN-1 operator
            Ugs = Ugs.get_upper_bound()

        p._Ugs = Ugs  # store for debugging purposes
        # Merge using the rule from Ehrgott2007 Prop 2
        Up_new = p.U.merge_upper_bounds(Ugs)

        # update the subproblem generator upper bound set
        p.U = Up_new

    def _reduce_lower_bound(self, p: Problem) -> None:
        # index of remaing subproblems not p
        S_hat = [p_other for p_other in self.P if p.name != p_other.name]
        other_lower_bounds = [p.L for p in S_hat]
        U_hat = reduce(
            lambda U1, U2: SearchArea(U1 + U2).get_upper_bound(), other_lower_bounds
        )

        self._set_global_lower_bound()
        # define polygon
        # Ugs = SearchArea(self.U+(-L_hat))
        A = SearchArea((self.L + (-U_hat)))

        p.L = p.L.merge_lower_bounds(A.get_lower_bound())

    # modify p.U for p in P
    def _reduce_upper_bounds(self, iteration: int = 0) -> None:
        """
        update each upper bound p.U using other subproblem bounds

        :param P: set of subproblems
        :param U: global upper bound used to reduce
        :param strategy: 'all' reduces with global bound. 'pairs' reduces using pairs of subproblems. updating strategy, choice of subproblem to update and using which other subproblems
        """

        match self.strategy_reduction:
            case "none":
                pass
            case "all":
                self._set_global_upper_bound()  # updates self.U
                for p in self.P:
                    self._reduce_upper_bound(p=p)
            case int():
                if iteration % self.strategy_reduction == 0:
                    self._set_global_upper_bound()  # updates self.U
                    for p in self.P:
                        self._reduce_upper_bound(p=p)
            case "first":
                if iteration == 0:
                    self._set_global_upper_bound()  # updates self.U
                    for p in self.P:
                        self._reduce_upper_bound(p=p)
            case _:
                raise NotImplemented

    def select_subproblem(self, iteration: int) -> Problem | None:

        if all((p.T.T.empty() for p in self.P)):
            return None  # terminate algorithm

        unsolved_subproblems = [p for p in self.P if not p.T.T.empty()]
        match self.strategy_subproblem_selection:
            case "random":
                # select random subproblem
                # return random.choice([p for p in self.P if not p.T.T.empty()])
                return random.choice(unsolved_subproblems)
            case "alternating":
                # using iteration - alternate between the non-solved subproblems in S
                return unsolved_subproblems[iteration % len(unsolved_subproblems)]
                # return self.P[iteration % len(self.P)]
            case "largest":
                # return largest search area - over all problems
                raise NotImplemented
            case "mostReduced":
                # return the node which have had the most area removed - larger potential for skipping non-generating solutions
                raise NotImplemented
            case "sequential":
                # return [p for p in self.P if not p.T.T.empty()][0]
                return unsolved_subproblems[0]
            case _:
                raise NotImplemented
        return

    def _update_node_priority(self, iteration: int):
        """method for updating the search tree for subproblems based on strategy self.strategy_update_node_priority"""
        # TODO: CALL Problem.update_search_tree_priority() - just naive implementation, creating the tree from scratch each time

    def get_minimum_generator(self):
        """sets variable p.Y_reduced for each p in self.P as the set of points in the reduced generator set. i.e. the set of subproblem points which help generate at least one solution of Yn"""

        if sum(len(p.Yn) for p in self.P) == 0:
            for p in self.P:
                p.Y_mgs = SolutionList()
            return
        Y_list = [p.Yn.as_pointlist() for p in self.P]

        msp = MinkowskiSumProblem(Y_list)
        mgs, Yn = algorithm2(msp)

        for Yns, Ygs in zip(msp.Y_list, mgs.Y_list):
            if __debug__:
                print(f"Yns: {len(Yns)}, Ygs: {len(Ygs)}")

        for p, Y_mgs in zip(self.P, mgs.Y_list):
            p.Y_mgs = SolutionList([y for y in p.Yn if AsmoPoint(y.val) in Y_mgs])

    def get_reduced_generator(self):

        Y_list = [p.Yn.as_pointlist() for p in self.P]
        self.reduced_generator = Y_list_to_fixed_reduced(Y_list)[-1]

    def solve(self, plotter: None | Plotter = None, time_limit=None):
        """[main algorithm]"""

        REDUCE_GENERATOR_BOUNDS = True

        # plot functions
        def reset_plots():
            for p in self.P:
                plotter.axs[f"main"].cla()
                plotter.axs[f"1_{p.name}"].cla()
                plotter.axs[f"2_{p.name}"].cla()
                plotter.axs[f"3_{p.name}"].cla()
                # plotter.axs[f"1_{p.name}"].set_title(f"Subproblem")
                plotter.axs[f"1_{p.name}"].legend()
                plotter.axs[f"2_{p.name}"].legend()
                plotter.axs[f"3_{p.name}"].legend()
            plotter.axs[f"main"].legend()

            # all_sp_points = [y.geom for p in self.P for y in p.Yse]
            # zoom = Polygon([y.geom for y in all_sp_points]).buffer(1000)
            for s, p in enumerate(self.P):
                p.color = s + 2
                # plotter.zoom(zoom, ax=f"1_{p.name}")
                # plotter.zoom(zoom, ax=f"2_{p.name}")
                # plotter.zoom(zoom, ax=f"3_{p.name}")

                # plotter.axs[f"2_{p.name}"].xlim = current + 10%
                plotter.axs[f"1_{p.name}"].set_title(f"Subproblem {s}")
                plotter.axs[f"2_{p.name}"].set_title(f"Search Node")
                plotter.axs[f"3_{p.name}"].set_title(
                    f"New refined generator bound sets"
                )

        def plot_search_areas():
            # return
            plotter.plot(
                self.Yn, name="$Y_N$", ax=f"main", color="gray", s=4, marker="s"
            )

            plotter.plot(
                self.hY, name="$\hat{Y}$", ax=f"main", color="green", s=3, marker="s"
            )
            # plotter.plot(self.Y,ax=f"main", color="green", linestyle='dashed')
            plotter.plot(
                self.U, name="$U$", ax=f"main", color="red", linestyle="dashed"
            )

            for p in self.P:

                plotter.plot(
                    p.L,
                    name="$L^s$",
                    ax=f"1_{p.name}",
                    color="blue",
                    linestyle="dashed",
                )
                plotter.plot(
                    p.U, name="$U^s$", ax=f"1_{p.name}", color="red", linestyle="dashed"
                )
                for n in p.T.T.queue:
                    pass

                    box_color = "gray" if not n.check_if_empty(p.L, p.U) else "red"
                    plotter.plot(n.R, ax=f"1_{p.name}", color=box_color)
                    plotter.plot(n.R, ax=f"2_{p.name}", color=box_color)

                plotter.plot(
                    p.Yn,
                    name="$Y_N^s$",
                    ax=f"1_{p.name}",
                    color="gray",
                    s=3,
                    marker="s",
                    zorder=100,
                )
                plotter.plot(
                    p.Yn,
                    name="$Y_N^s$",
                    ax=f"2_{p.name}",
                    color="gray",
                    s=3,
                    marker="s",
                    zorder=100,
                )
                plotter.plot(
                    p.Y_mgs,
                    name="$Y_g^s$",
                    ax=f"1_{p.name}",
                    color="black",
                    s=1,
                    marker="s",
                    zorder=100,
                )
                plotter.plot(
                    p.Y_mgs,
                    name="$Y_g^s$",
                    ax=f"2_{p.name}",
                    color="black",
                    s=1,
                    marker="s",
                    zorder=100,
                )
                plotter.plot(
                    p.hY,
                    name="$\hat{Y}^s$",
                    ax=f"1_{p.name}",
                    color="green",
                    marker="s",
                    s=2,
                    zorder=101,
                )

                # add buffer to make room for the legend
                zoom1 = Polygon([y.geom for y in p.Yn])
                if zoom1:
                    plotter.zoom(
                        zoom1, ax=f"1_{p.name}", buffer=35, buffer_relative=True
                    )
                    plotter.zoom(
                        zoom1, ax=f"3_{p.name}", buffer=35, buffer_relative=True
                    )
                if False:  # plot L_hat
                    L_hat = self._get_L_hat(p)
                    local_nadir_points = induced_UB(p.hY.as_pointlist())
                    for u in local_nadir_points.points:
                        u_minus_L = (-L_hat) + AsmoPoint(u.val)
                        plotter.plot(u_minus_L, name="$\hat{L}^s$", ax=f"1_{p.name}")
                    plotter.plot(
                        L_hat,
                        name="$\hat{L}^s$",
                        ax=f"main",
                        linestyle="dashed",
                    )

                # add legend
                plotter.axs[f"1_{p.name}"].legend(loc="upper right")
                plotter.axs[f"2_{p.name}"].legend(loc="upper right")
                plotter.axs[f"3_{p.name}"].legend(loc="upper right")

            plotter.axs[f"main"].legend(loc="upper right")

        # initialization step
        for p in self.P:
            # set initial p.Yse, p.hY, p.U, p.L, p.T
            p.initialize_subproblem()
            self.statistics["IP-calls"] += (
                len(p.Yse) * 2
                - 1
                # 0  # initial IP calls to find supported solutions
            )
            self.statistics["nodes explored"] += 1
            # self.statistics["nodes explored"] += 0

        # set global upper bound
        self._set_global_upper_bound()  # U = ND_SUM(P[0].U + ... + P[S].U)

        if plotter:
            for ps in self.P:
                plotter.plot(
                    ps.U,
                    name="$U^s$",
                    ax=f"3_{ps.name}",
                    color="red",
                    linestyle="dashed",
                )
                # plotter.plot(ps.L,ax=f"3_{ps.name}", color="blue", linestyle='dashed')

            plot_search_areas()
            plotter.save("tests/test_asmo_solver/0.pdf")
        # reduce upper bounds
        if REDUCE_GENERATOR_BOUNDS:
            self._reduce_upper_bounds()  # updates p.U for each p in P

        # partition the search areas for each subproblem
        # initate search tree for each subproblem
        for p in self.P:
            p.partition_search_area(p.hY)
            # print(f"Initial branch tree size: {len(p.T.T.queue)=}")
            # print(f"Initial branch tree size: {len(p.Yn)=}")
            # print(f"Initial branch tree size: {len(p.hY)=}")

        # initial plot
        if plotter:
            plot_search_areas()

            for ps in self.P:
                plotter.plot(
                    ps.U,
                    name="$(U_g^s \cup U^s)_N$",
                    ax=f"3_{ps.name}",
                    color="red",
                    linestyle="solid",
                )
                plotter.plot(
                    ps.L,
                    name="$L^s$",
                    ax=f"3_{ps.name}",
                    color="blue",
                    linestyle="solid",
                )
                plotter.axs[f"3_{ps.name}"].legend()

            plotter.save("tests/test_asmo_solver/0.pdf")
            reset_plots()

        # main loop
        i = 0
        time_start = time.time()
        while True:
            i += 1
            if i > 10000:
                print(f"max iteration count reached (TERMINATING-ERROR): {i=}")
                raise TimeoutError
                break

            if time_limit and time.time() - time_start > time_limit:
                print(f"max iteration count reached (TERMINATING-ERROR): {i=}")
                raise TimeoutError
                # return False

            # select node - subproblem and search area
            # print(f"iteration: {i=}")

            if plotter:  # plot the chosen node
                reset_plots()
                plotter.axs[f"main"].set_title(f"iteration {i}")
                plot_search_areas()

                for s, p in enumerate(self.P):
                    plotter.axs[f"1_{p.name}"].set_title(
                        f"Subproblem {s+1}:  search nodes = {len(p.T)}"
                    )

            p = self.select_subproblem(
                iteration=i
            )  # using chosen strategy self.strategy_subproblem_selection

            if p is None:
                break  # terminate, all subproblems solved

            assert (
                len(p.T.T.queue) > 0
            ), f"selected subproblem {p.name} has no search tree nodes left to search"

            n = p.T.getNode()

            assert (
                n is not None
            ), f"no node found in search tree {p.T.T.queue} for subproblem {p.name}"

            f_2_text = "Seach node"
            # check for pruning
            if n.check_if_empty(p.L, p.U):
                print(f"SKIPPING empty node search area {n}")
                f_2_text += " empty => SKIPPED"
                if plotter:
                    plotter.axs[f"2_{p.name}"].set_title(f_2_text)
                    plotter.zoom(
                        n.R.geom, ax=f"2_{n.p.name}", buffer=20, buffer_relative=False
                    )
                    # plot bounds
                    plotter.plot(
                        n.p.L,
                        ax=f"2_{n.p.name}",
                        color="blue",
                        linestyle="dashed",
                        name="$\mathcal{L}^s$",
                    )
                    plotter.plot(
                        n.p.U,
                        ax=f"2_{n.p.name}",
                        color="red",
                        linestyle="dashed",
                        name="$\mathcal{U}^s$",
                    )
                    # plot n.R on 1_
                    plotter.plot(
                        n.R.geom,
                        ax=f"3_{n.p.name}",
                        color="orange",
                        name="search node",
                    )
                    plotter.save(f"tests/test_asmo_solver/{i}.pdf")
                i += 1
                continue

            # search brancing node and branch - updates n.p.hY and n.p.T
            p.search_node(n)
            self.statistics["IP-calls"] += n.statistics["IP-calls"]
            self.statistics["nodes explored"] += 1

            # add the newly found nodes to the search tree
            # for new_node in n.T:
            # p.T.addNode(new_node)

            #
            #     if False:
            #         y_ul = Solution((n.R.ul[0]-1, n.R.ul[1] + 1) , p_name = '_', x=np.empty(0))
            #         y_lr = Solution((n.R.lr[0]+1, n.R.lr[1] - 1) , p_name = '_', x=np.empty(0))
            #         p.partition_search_area( SolutionList([y_ul] + list(n.tY) + [y_lr]))
            #     # p.partition_search_area(list(n.tY))
            # else:
            #     p.partition_search_area(n.tY)
            #

            if plotter:
                plotter.plot(
                    n.R, name="$\eta$", ax=f"1_{n.p.name}", color="orange", alpha=1
                )
                plotter.plot(
                    n.R,
                    name="$\eta$",
                    ax=f"2_{n.p.name}",
                    color="orange",
                    # hatch="x",
                    alpha=1,
                )
                # plotter.plot(p.L,ax=f"2_{p.name}", color="blue", linestyle='dashed')
                plotter.plot(p.L, ax=f"3_{p.name}", color="blue", linestyle="dashed")
                # plotter.plot(p.U,name='$U^{s(\eta)}$',ax=f"2_{p.name}", color="red", linestyle='dashed')
                # plotter.zoom(
                #     n.R.geom, ax=f"2_{n.p.name}", buffer=20, buffer_relative=True
                # )
                plotter.plot(n.tL, name="$L^\eta$", ax=f"2_{n.p.name}", color="blue")
                plotter.plot(n.tU, name="$U^\eta$", ax=f"2_{n.p.name}", color="red")
                plotter.plot(
                    n.tY, name="$\hat{Y}^\eta$", ax=f"2_{n.p.name}", color="green"
                )
                plotter.axs[f"2_{n.p.name}"].legend()

                # plotter.plot(ps.L,ax=f"3_{ps.name}", color="blue", linestyle='dashed')

                # clear remaining
                for p_other in self.P:
                    if p_other != n.p:
                        plotter.axs[f"2_{p_other.name}"].cla()

                plotter.zoom(
                    n.R.geom, ax=f"2_{n.p.name}", buffer=10, buffer_relative=False
                )
                # if p.Yse:
                f_2_text += " $|\hat{Y}^\eta|=" + f"${len(n.tY)}"
                plotter.axs[f"2_{p.name}"].set_title(f_2_text)
            # update search area
            p._update_bounds_from_node(n)

            if plotter:  # plot updated bounds - after node
                for ps in self.P:
                    plotter.plot(
                        ps.U, ax=f"3_{ps.name}", color="red", linestyle="dashed"
                    )
                plotter.plot(n.R, ax=f"3_{n.p.name}", color="orange", alpha=1)
                plotter.save(f"tests/test_asmo_solver/{i}.pdf")

            # update global upper bound
            # self._update_global_upper_bound(y=None)
            # partition search space

            if p.search_method == "bbm" and n.tY:
                for new_node in n.T:
                    p.T.addNode(new_node)
            else:
                p.partition_search_area(n.tY)

            # refine bound sets
            if REDUCE_GENERATOR_BOUNDS:
                self._reduce_upper_bounds(iteration=i)

            if plotter:  # plot updated bounds - after refinement
                for ps in self.P:
                    # plotter.plot(ps.U,name='$U_g^s$',ax=f"3_{ps.name}", color="red", linestyle='solid')
                    plotter.plot(
                        ps.U,
                        name="$(U_g^s \cup U^s)_N$",
                        ax=f"3_{ps.name}",
                        color="red",
                        linestyle="solid",
                    )
                    plotter.plot(
                        ps.L,
                        name="$L^s$",
                        ax=f"3_{ps.name}",
                        color="blue",
                        linestyle="solid",
                    )
                    # Anew = SearchArea.get_search_area_geom(ps.L,ps.U)
                    # plotter.plot(Anew, ax=f"3_{ps.name}", name='A')

                    plotter.axs[f"3_{ps.name}"].legend()
                plotter.save(f"tests/test_asmo_solver/{i}.pdf")

            # if i>18:
            # break
        print_timeit()
        self._update_global_upper_bound(None)
        return True
