# public library imports
from __future__ import annotations  # allow annotation self references (eg. in KD_Node)
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import csv
import json
import os
import collections
from functools import partialmethod
import matplotlib

matplotlib.use("TKAgg")

"""
Class
@Point

y1 = Point((4,2))
y2 = Point([3,2])

y1 < y2
>> False
t2 <= y1 
>> True


Class
@PointList 

Y1 = PointList((y1, y2))
Y2 = PointList((y2))
Y3 = PointList.from_csv(fname)

Y1 == Y2
>> False, since counter is off Y1:{y1: 1, y2: 1}, while Y2: {y2: 1}

Y3.save_csv(fname)
>> saves the list to a csv file with name fname

Y1.plot()
> plots set of points, if True plt.show() is called

Y2.dominates_point(y1)
>> True if the point y1 is dominated by the set Y2


"""


@dataclass
class Point:
    """Point. A vector object used as elements of PointList(s). Are equipped with componenwise relations <,<=, plot(self) for visualization.

    Example(s):
        Point((2,3))
        Point((2,3,1))
    """

    val: np.array(iter)
    dim = None
    plot_color = None
    cls = None

    def __post_init__(self):
        if not isinstance(self.val, np.ndarray):
            self.val = np.array(self.val)
        if self.dim == None:
            self.dim = len(self.val)
        self.val = (self.val).round(decimals=4)  # round anything

    def __lt__(self, other: Point):
        """__lt__. return True if self dominates other (componen-wise) minimization sense

        Args:
            other (Point): other
        """
        if all(self.val == other.val):
            return False
        return all(self.val <= other.val)

    def __le__(self, other):
        return all(self.val <= other.val)

    def le_d(self, other, d: int):
        return all((self.val[p] <= other.val[p] for p in range(d)))

    def lt_d(self, other, d: int):
        if all((self.val[p] == other.val[p] for p in range(d))):
            return False
        return all((self.val[p] <= other.val[p] for p in range(d)))

    def strictly_dominates(self, other):
        return all(self.val < other.val)

    def lex_le(self, other):
        if len(self.val) == 2:
            if self.val[0] > other.val[0]:
                return False
            if self.val[0] < other.val[0]:
                return True
            if self.val[0] == other.val[0] and self.val[1] > other.val[1]:
                return False
            else:
                return True
        if len(self.val) > 2:
            for p in range(self.dim):
                if self[p] < other[p]:
                    return True
                elif self[p] > other[p]:
                    return False
            return True

    def __gt__(self, other):
        if all(self.val == other.val):
            return False
        return all(self.val >= other.val)

    def __iter__(self):
        return self.val.__iter__()

    def __hash__(self):
        return tuple(self.val).__hash__()

    def __eq__(self, other):
        return (self.val == other.val).all()

    def is_close(self, other):
        return np.isclose(self.val, other.val).all()

    def __repr__(self):
        return tuple((float(vi) for vi in self.val)).__repr__()

    def __getitem__(self, item):
        return self.val[item]

    def __add__(self, other):
        if isinstance(other, Line):
            return other + self
        if isinstance(other, PointList):
            return PointList((self,)) + other

        return Point(self.val + other.val)

    def __sub__(self, other):
        return Point(self.val - other.val)

    def __mul__(self, other):
        if isinstance(other, int):
            new_point = Point(self.val * other)
        elif isinstance(other, float):
            new_point = Point(self.val * other)
        elif isinstance(other, Point):
            new_point = Point(self.val * other.val)
        else:
            raise TypeError(f"__mul__ not implemented for {type(other)=}")
        new_point.cls = self.cls
        return new_point

    def plot(
        self,
        SHOW=False,
        fname=None,
        ax=None,
        l=None,
        label_only=False,
        color=None,
        **kwargs,
    ):
        assert self.dim <= 3, "Not implemented for p > 3"
        ax = ax if ax else plt
        color = color if (color is not None) else self.plot_color
        kwargs["color"] = color
        if self.dim == 3:
            ax.scatter = ax.scatter3D

        if not label_only:
            plot = ax.scatter(*self.val, **kwargs)
            self.plot_color = plot.get_facecolor()
        if l != None:
            if self.dim == 3:
                ax.text(*self.val, l)
            else:
                ax.annotate(text=l, xy=self.val, xytext=self.val * 1.02)

        if l != None:
            ax.legend(loc="upper right")
        if fname:
            ax.savefig(fname, dpi=200)
            ax.cla()
        if SHOW:
            ax.show()
        return ax

    def plot_cone(
        self, ax=None, quadrant=1, y_nadir: Point = None, color="darkgray", **kwargs
    ):
        assert self.dim <= 2, "plot_cone Not implemented for p > 2"
        ax = ax if ax else plt
        color = color if (color is not None) else self.plot_color
        kwargs["color"] = color
        kwargs["linewidth"] = (
            0 if ("linewidth" not in kwargs) else kwargs["linewidth"]
        )  # default to linewidth = 0
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        if y_nadir:
            xmax = y_nadir[0]
            ymax = y_nadir[1]
        if quadrant == 1:
            ax.add_patch(
                Rectangle(
                    (self[0], self[1]),
                    xmax - self[0],
                    ymax - self[1],
                    fill=False,
                    hatch="xx",
                    **kwargs,
                )
            )
        if quadrant == 2:
            ax.add_patch(
                Rectangle(
                    (xmin, self[1]),
                    self[0] - xmin,
                    ymax - self[1],
                    fill=False,
                    hatch="xx",
                    **kwargs,
                )
            )
        if quadrant == 3:
            ax.add_patch(
                Rectangle(
                    (xmin, ymin),
                    self[0] - xmin,
                    self[1] - ymin,
                    fill=False,
                    hatch="xx",
                    **kwargs,
                )
            )
        if quadrant == 4:
            ax.add_patch(
                Rectangle(
                    (self[0], 0), xmax - self[0], ymax, fill=False, hatch="xx", **kwargs
                )
            )
        return ax


@dataclass
class PointList:
    """
    A class used to represent a set of Points

    ...

    Attributes
    ----------
    points: iter[Point]
        an iterable containing a set of points
    dim : str
        the dimension of the points
    plot_color : str
        color used when plotted using matplotlib, initially None
    statistics: dict
        a dictionary containing statistics for the PointList. This is updated when the PoinsList is return by several methods.

    Methods
    -------
    __add__(self, other)
        returns the Minkowski sum of the two pointlists.
        defined as Y1-Y2 = {y1+y2: for y1 in Y1, for y2 in Y2}

    __eq__(self,other)
        returns true if the two pointlist contains the same (and same amount of) points. Other attributes are ignored

    __getitem__(i)
        returns the Point at index i, to support slicing

    __sub__(self, other)
        returns the Minkowski difference of the two pointlists. Defined as Y1-Y2 = {y1-y2: for y1 in Y1, for y2 in Y2}

    __mul__(self, other)
        returns the (Minkowski) product of the two pointlists. Defined as Y1*Y2 = {y1*y2: for y1 in Y1, for y2 in Y2}

    as_dict(self)
        returns a dictionary version of the PointList object

    as_np_array(self)
        returns an np.array containg all points

    dominates(other)
        returns true of the pointlist dominates other. Use params for weakly,strict dominance

    dominates_point(y:Point)
        returns true if some point of the PointList dominates the point y

    from_csv(path)
        returns a PointList based on the file path, Only points are read no attributes

    from_json(path)
        returns a PointList based on the file path

    from_raw(path)
        returns a PointList based on the file path, only points are read no attributes. See save_raw for file description

    get_ideal()
        returns the ideal point of the set. Component-wise min point.

    get_nadir()
        returns the nadir point of the set. Component-wise max point.

    plot(l = 'LABEL', SHOW=True)
        plots the set of points contained in PointList

    print_data()
        prints the PointList

    save_csv(filepath)
        saves the pointlist in a csv format. ONLY points are saved, no statistics.

    save_json(filepath)
        saves the pointlist in a json format. Uses the as_dict method

    save_raw(filepath)
        saves the pointlist in a raw format, these files are slightly more memory efficient and can be read by the C NonDomDC filter

    weakly_dominates_point(y:point)
        checks if the PointList weakly dominates the point y
    """

    points: tuple[Point] = ()
    dim = None
    plot_color = None
    statistics: dict = None
    filename = None
    np_array: np_array = None

    def __post_init__(self):
        # Check if SINGLETON: allows for PointList((y)) where y is of class Point
        if isinstance(self.points, Point):
            self.points = (self.points,)
        else:  # unpack list
            self.points = tuple([Point(y.val) for y in self.points])
            # self.points = tuple([y if isinstance(y, Point|Line)  else Point(y) for y in self.points])
        if self.points:
            self.dim = self.points[0].dim

        self.statistics = {
            "p": [self.dim],
            "card": [len(self.points)],
            "supported": [None],
            "extreme": [None],
            "unsupported": [None],
            "min": [None],
            "max": [None, None],
            "width": [None, None],
            "method": [None],
        }

    def __iter__(self) -> tuple[Point]:
        return tuple(self.points).__iter__()

    def iter_endpoints(self) -> iter[Point]:
        for l in self.points:
            assert isinstance(l, Line)
            yield l[0]
            yield l[1]

    def __len__(self):
        return tuple(self.points).__len__()

    def __repr__(self):
        return f"PointList{self.points}"

    def plot(
        self,
        l=None,
        SHOW=False,
        fname=None,
        ax=None,
        line=False,
        lines_only=False,
        color=None,
        point_labels=False,
        **kwargs,
    ):

        ax = ax if ax else plt
        if len(self.points) is None:
            print("WARNING: trying to plot empty PointList... Skipping")
            return
        assert self.dim <= 3, "Not implemented for p > 3"
        # color = self.plot_color if (color is not None) else color
        color = color if (color is not None) else self.plot_color
        kwargs["color"] = color

        if self.dim == 3:
            ax.scatter = ax.scatter3D
            ax.plot = ax.plot3D

        if line:
            plot = ax.plot(*zip(*self.points), label=l, **kwargs)
            self.plot_color = plot[-1].get_color()
        else:
            points = []
            lines = []
            for y in self.points:
                # print(f"{y=}")
                if isinstance(y, Point):
                    points.append(y)
                elif isinstance(y, Line):
                    points.append(y[0])
                    points.append(y[1])
                    lines.append(y)
                else:
                    print(f"{y, type(y)}")
                    raise NotImplementedError
            # print(f"{points=}")
            # print(f"{lines=}")
            if points:
                if not lines_only:
                    plot = ax.scatter(*zip(*points), label=l, **kwargs)
                    self.plot_color = plot.get_facecolors()
                    kwargs["color"] = self.plot_color
            for y in lines:
                plot = ax.plot(*zip(*y.points), **kwargs)
                self.plot_color = plot[-1].get_color()
                kwargs["color"] = self.plot_color
            # self.plot_color = plot.to_rgba(-1) # save used color to object
        if l:
            ax.legend(loc="upper right")
        if fname:
            plt.savefig(fname, dpi=200)
            plt.cla()
        if point_labels:
            if point_labels == True:
                # point_labels = ["$y^{" +  f"{i}" + "}$" for i, _ in enumerate(self, start = 1)]
                pass
            # add labels to points
            for i, y in enumerate(self):
                # y.plot(ax = ax, l= "$y^{" +  f"{i}" + "}$", label_only=True)
                if isinstance(y, Line):
                    point_label = "$l^{" + f"{i+1}" + "}$"
                    (y[0] * 0.5 + y[1] * 0.5).plot(
                        ax=ax, l=point_label, label_only=True
                    )
                else:
                    point_label = "$y^{" + f"{i+1}" + "}$"
                    y.plot(ax=ax, l=point_label, label_only=True)

        if SHOW:
            ax.show()

        return ax

    def dominates_point(self, point: Point):
        for y in self.points:
            if y < point:
                # return True
                return y
        return False

    def weakly_dominates_point(self, point: Point):
        for y in self.points:
            if y <= point:
                return y
                # return True
        return False

    def __add__(self, other: Point):
        """__add__. returns (PointList) with Minkowski sum of the two pointlists. Defined as Y1+Y2 = {y1+y2: for y1 in Y1, for y2 in Y2}
        Args:
            other (PointList): PointList
        Returns:
            (PointList) with Minkowski sum
        """
        return PointList([y1 + y2 for y1 in self for y2 in other])

    def __sub__(self, other: PointList):
        """__sub__. returns (PointList) with Minkowski difference of the two pointlists. Defined as Y1-Y2 = {y1-y2: for y1 in Y1, for y2 in Y2}
        Args:
            other (PointList): PointList
        Returns:
            (PointList) with Minkowski difference
        """
        return PointList([y1 - y2 for y1 in self for y2 in other])

    def __mul__(self, other: PointList):
        """__mul__. returns (PointList) with Minkowski product of the two pointlists. Defined as Y1*Y2 = {y1*y2: for y1 in Y1, for y2 in Y2}
        Args:
            other (PointList): PointList
        Returns:
            (PointList) with Minkowski product
        """
        match other:
            case float() | int() | Point():
                return PointList([y * other for y in self])
            case _:
                print(f"{other=}")
                print(f"{type(other)=}")
                raise NotImplementedError

    def get_nadir(self):
        """get_nadir. Returns the nadir point of the set. Component-wise max point.

        Returns:
            (Point) nadir point of the PointList
        """
        nadir_vals = list(tuple(self.points[0].val))
        for point in self.points:
            for p in range(self.dim):
                if nadir_vals[p] < point.val[p]:
                    nadir_vals[p] = point.val[p]
        self.nadir = Point(nadir_vals)
        return self.nadir

    def get_ideal(self):
        """get_ideal. Returns the nadir point of the set. Component-wise min point.

        Returns:
            (Point) ideal point of the PointList
        """
        ideal_vals = list(tuple(self.points[0].val))
        for point in self.points:
            for p in range(self.dim):
                if ideal_vals[p] > point.val[p]:
                    ideal_vals[p] = point.val[p]
        self.ideal = Point(ideal_vals)
        return self.ideal

    def dominates(self, other, power="default"):
        """dominates. Returns true of the PointList dominates other. Use params for weakly,strict dominance

        Args:
            other:
            (str) power: "default", "weakly", "strict"
        Returns:
            (bool) true if PointList self dominates other wrt. power
        """
        match power:
            case "default":
                if self == other:
                    return False
                for y in other.points:
                    if any((l <= y for l in self.points)):
                        continue
                    else:
                        return False
                return True

            case "strict":
                for y in other.points:
                    if any((l < y for l in self.points)):
                        continue
                    else:
                        return False
                return True

    def save_csv(self, filename: str):
        """save_csv. Saves the PointList in a csv format. ONLY points are saved, no statistics.
        Args:
            filename (str): filename
        """
        with open(f"{filename}", "w") as out:
            csv_out = csv.writer(out)
            for y in self.__iter__():
                csv_out.writerow(y)

    def save_raw(self, filename: str):
        """save_csv. Saves the PointList in a raw format. ONLY points are saved, no statistics. Used by C interface, and more memory efficient than json/csv.
        Args:
            filename (str): filename
        """
        try:
            with open(filename, "w") as out:
                out.write(f"{self.statistics['p'][0]}" + "\n")
                out.write(f"{self.statistics['card'][0]}" + "\n")
                for y in self.__iter__():
                    out.write(" ".join([f"{yp:.6f}" for yp in y]) + "\n")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not save raw file to {filename}, directory does not exist. {os.getcwd()}"
            ) from e

    def from_raw(filename: str):
        """from_raw. Returns a PointList based on the file path, only points are read no attributes. See save_raw for file description

        Args:
            filename (str): filename
        Returns:
            (PointList) PointList read from filename
        """
        # raw format used in c-interface
        with open(filename, "r") as rawfile:
            dim = int(rawfile.readline())
            n = int(rawfile.readline())
            lines = rawfile.read().splitlines()
        y_list = []
        for line in lines:
            y = Point([float(yp) for yp in line.split(" ") if yp != ""])
            y_list.append(y)
        return PointList(y_list)

    def from_csv(filename: str):
        with open(f"{filename}", "r") as csvfile:
            points = []
            for y in csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC):
                points.append(Point(y))
            return PointList(points)

    def as_dict(self):
        """as_dict. returns a dictionary version of the PointList object
        Returns:
            (PointList) a dictionary containing the PointList points and statistics.
        """

        PointList_dict = {
            "points": [
                dict(
                    {f"z{p+1}": float(point[p]) for p in range(point.dim)},
                    **({"cls": point.cls}),
                )
                for point in self.points
                if isinstance(point, Point)
            ],
            "lines": [
                dict({f"y{p+1}": tuple(point[p]) for p in range(2)})
                for point in self.points
                if isinstance(point, Line)
            ],
            # "lines": [dict({f"l{p+1}": str(point)},**({'cls':point.cls})) for point in self.points if isinstance(point, Line)],
            "statistics": self.statistics,
        }

        #         PointList_dict = {
        # "points":
        # [dict({f"z{p+1}": float(point[p]) for p in range(point.dim)},**({'cls':point.cls})) for point in self.points],
        # 'statistics': self.statistics
        # }
        return PointList_dict

    def as_np_array(self):
        """as_np_array. returns an np.array with the points in PointList
        Returns:
            (np.array) containing the list of points.
        """
        return np.array([y.val for y in self.points])

    def save_json(self, filename: str, max_file_size: int = 100):
        """save_json. Saves the pointlist in a json format. Uses the as_dict method

        Args:
            filename (str): filename
            max_file_size (int): max_file_size in GB
        """

        json_str = json.dumps(self.as_dict(), indent=None, separators=(",", ":"))
        # Calculate size (approx) in bytes
        size_mb = len(json_str.encode("utf-8")) / 1_000_000
        if size_mb >= max_file_size:
            if True:
                print(
                    f"*** {filename}, pointlist with {len(self)} points too large for json. Saving raw format. ESTIMATED MB {size_mb} > {max_file_size}=MAX"
                )
                self.save_raw(filename.replace(".json", ".raw"))
                # return
            if True:
                # print(f"*** {filename}, pointlist with {len(self)} points too large for json. Saving json without points. ESTIMATED MB {size_mb} > {max_file_size}=MAX")
                json_str = self.as_dict()
                json_str["points"] = []
                json_str = json.dumps(json_str, indent=None, separators=(",", ":"))
        with open(filename, "w") as json_file:
            json_file.write(json_str)
            # json.dump(self.as_dict(), json_file)

    def from_json_str(json_dict: dict) -> PointList:
        """from_json_str. Reads the PointList from a str containing a dictionary version of a PointList

        Args:
            json_dict (dict): json_dict, with the PointList object

        Returns:
            PointList:
        """
        statistics = json_dict["statistics"]
        points = []
        for json_point in json_dict["points"]:
            values = [json_point[f"z{p+1}"] for p in range(statistics["p"][0])]
            # TODO: Error if values not casted to float65  - does not work for int64? <08-02-24> #
            values = np.float64(values)
            values = np.round(values, 1)  # round to nearest 5th decimal place
            point = Point(values)
            if "cls" in json_point:
                point.cls = json_point["cls"]
            else:
                point.cls = None
            points.append(point)

        lines = []
        if json_dict.get("lines"):
            for l_str in json_dict["lines"]:
                # print(f"{l_str=}")
                # print(f"{l_str['y1']=}")
                # print(f"{[float(zi) for zi in l_str['y1']]=}")

                y1 = Point(np.array(l_str["y1"]))
                y2 = Point(np.array(l_str["y2"]))
                # print(f"{y1,y2=}")
                lines.append(Line((y1, y2)))

        Y = PointList(points + lines)

        Y.statistics = statistics

        return Y

    def from_json(filename: str):
        print(filename)
        try:
            with open(filename, "r") as json_file:
                json_dict = json.load(json_file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not load json file from {filename}, file does not exist. {os.getcwd()}. Found only {os.listdir()}"
            )
        return PointList.from_json_str(json_dict)

    def print_data(self):
        N_POINTS = len(self.points)
        print(f"{N_POINTS=}")

    def __eq__(self, other):
        return collections.Counter(self.points) == collections.Counter(other.points)

    def __lt__(self, other):
        """
        input: two PointLists
        output: return True if each point of other is dominated by at least one point in self
        """
        for y2 in other:
            for y1 in self:
                if y1 < y2:
                    break
            else:  # finally, if for loop finishes normaly
                return False
        return True

    def __getitem__(self, subscript):
        result = self.points.__getitem__(subscript)
        if isinstance(subscript, slice):
            return PointList(result)
        else:
            return result

    def removed_duplicates(self):
        """
        returns a PointList with all duplicates removed
        OBS: all statistics are reset
        """
        return PointList(set(self.points))

    def get_left_most_dominator(self, y):
        """returns the dominating point with the smallest second coordinate - used to update epsilon"""

        left_most_dominator = None
        for l in self:
            u_dom = l.dominates_point(y)
            if u_dom:
                if not left_most_dominator:
                    left_most_dominator = u_dom
                elif u_dom[0] < left_most_dominator[0]:
                    left_most_dominator = u_dom
        return left_most_dominator


@dataclass
class MinkowskiSumProblem:
    Y_list: tuple[PointList]
    filename: str = None
    dim: int = None
    S: int = None
    sp_filenames: list = None

    def __post_init__(self):
        self.S = len(self.Y_list)

    def __iter__(self):
        return self.Y_list.__iter__()

    def from_json(filename: str, as_filename=True):
        with open(filename, "r") as json_file:
            json_list = json.load(json_file)
            json_dict = json_list[0]
            if len(json_list) > 1:
                statistics = json_list[1]
            else:
                statistics = None

        Y_list = []
        sp_filenames = []
        for V, Y_filename in json_dict.items():
            if as_filename:
                Y = Y_filename
            elif isinstance(Y_filename, str):
                print(f"{Y_filename}")
                Y = PointList.from_json("instances/" + Y_filename)
                Y.filename = Y_filename
            else:
                Y = PointList.from_json_str(Y_filename)
            Y_list.append(Y)
        MSP = MinkowskiSumProblem(Y_list)
        MSP.statistics = statistics
        MSP.filename = filename
        try:
            MSP.dim = Y_list[0].dim
        except AttributeError as e:
            pass

        return MSP

    def save_json(self, filename):
        out_dict = [
            {f"V{s}": Y.as_dict() for s, Y in enumerate(self.Y_list, start=1)},
            self.statistics,
        ]
        json_str = json.dumps(out_dict, indent=1, separators=(",", ":"))
        with open(filename, "w") as json_file:
            json_file.write(json_str)

    def from_subsets(filenames: iter[str]):
        Y_list = []
        sizes = "|"
        method = ""
        for Y_filename in filenames:
            Y = PointList.from_json("./instances/subproblems/" + Y_filename)
            Y_list.append(Y)
            sizes += Y_filename.split("-")[2] + "|"
            method += Y_filename.split("-")[3].split(".")[0]
        filename = f"MSP-special-{sizes}-{method}"
        MSP = MinkowskiSumProblem(Y_list)
        MSP.filename = filename
        MSP.dim = Y_list[0].dim
        return MSP

    def __repr__(self):
        string = f"MSP( filename={self.filename.split('/')[-1]}, dim={self.dim}, "
        for s, Y in enumerate(self.Y_list):
            string += f"|Y{s+1}|={len(Y)} "

        string += ")"
        return string

    def plot(
        self, hidelabel=False, set_label=r"\mathcal{Y}_{\mathcal{N}}", ax=None, **kwargs
    ):
        ax = ax if ax else plt
        for s, Y in enumerate(self.Y_list):
            Y.plot(
                l="_" * hidelabel + "$ " + set_label + "^{" + str(s + 1) + "}$",
                ax=ax,
                **kwargs,
            )


@dataclass
class Line(PointList):
    """Line. A 2d line representing the convex hull of two points ie a line segment . Are equipped with componenwise relations <,<=, plot(self) for visualization.

    Example(s):
        Line((2,3),(1,4))
        Point(y1, y2) for y1, y2 of class Point
    """

    points: iter[Point]
    plot = partialmethod(PointList.plot, line=True)
    y1: Point = None
    y2: Point = None
    a: float = None
    b: float = None

    def __post_init__(self):
        self.dim = 2
        y1 = self.points[0]
        y2 = self.points[1]
        a = (y2[1] - y1[1]) / (y2[0] - y1[0])
        b = y1[1] - a * y1[0]
        self.is_vertical = True if np.isclose(y2[0], y1[0]) else False
        # print(f"{b=}")
        # print(f"{y2[1] - a*y2[0]}")
        assert np.isclose(b, y2[1] - a * y2[0])
        # if not self.is_vertical:
        # assert y1[0] <= y2[0] # assert the line is in order

        assert not y1.is_close(y2)
        self.y1, self.y2 = y1, y2
        self.a, self.b = a, b

    def plot_line(self, ax=None, **kwargs):
        ax = ax if ax else plt
        ax.axline((0, self.b), slope=self.a, **kwargs)

    def __hash__(self):
        return tuple(self.points).__hash__()

    def plot_cone(self, ax=None, y_nadir: Point = None, color="darkgray", **kwargs):
        assert self[0][0] < self[1][0]
        assert self.dim <= 2, "plot_cone Not implemented for p > 2"
        color = color if (color is not None) else self.plot_color
        ax = ax if ax else plt
        kwargs["color"] = color
        kwargs["linewidth"] = (
            0 if ("linewidth" not in kwargs) else kwargs["linewidth"]
        )  # default to linewidth = 0

        if not y_nadir:
            ymin, ymax = ax.get_ylim()
            xmin, xmax = ax.get_xlim()
            y_nadir = Point((xmax, ymax))
        # if local_nadir:
        # assert not y_nadir, 'incompatible y_nadir and local_nadir options'

        # if y_nadir:
        # xmax = y_nadir[0]
        # ymax = y_nadir[1]
        # print(f"{self[0].val}")
        # print(f"{self[1].val}")
        # print(f"{y_nadir.val}")
        elif isinstance(y_nadir, str) and y_nadir == "search_area":
            y_nadir = self.get_nadir()
        dominance_cone = patches.Polygon(
            (
                (self[0]).val,
                self[1].val,
                (y_nadir[0], self[1][1]),
                y_nadir.val,
                (self[0][0], y_nadir[1]),
            ),
            closed=False,
            hatch="xx",
            edgecolor=color,
            fill=False,
            **kwargs,
        )
        ax.add_patch(dominance_cone)
        # ax.add_patch(Rectangle((self[0], 0), xmax - self[0], ymax, fill=False, hatch='xx', **kwargs))

    def length(self):
        return np.linalg.norm(self.y1.val - self.y2.val)

    def eval(self, x):
        # print(f"y1 = {self.y1[0]=} <= {x} = x : {self.y1[0] <= x}")
        # print(f"y2 = {self.y2[0]=} >= {x} = x : {self.y1[0] <= x}")

        # print(f"{self.y1,self.y2=}")
        # assert self.y1[0] <= x
        # assert x <= self.y2[0], (x,self.y2[0])
        return self.a * x + self.b

    def dominates_point(self, y, return_segment=False):
        """returns the left-most point of the line which dominates y else None"""
        # if self.y2 < y: return self.y2
        if self.y1 < y:
            return self.y1
        # assert self.y1[0] <= self.y2[0] # assert the line is in order
        # if self.y1[0] <= self.y2[0]: return Line.dominates_point(Line((self[1],self[0])),y)
        if y[0] < self.y1[0]:
            return None
        if y[1] < self.y2[1]:
            return None
        # case: y is in the rectangle defined by y1 and y2

        if self.eval(y[0]) < y[1]:
            y_ul = Point(((y[1] - self.b) / self.a, y[1]))
            if y_ul.is_close(y):
                return None

            if return_segment == False:
                return y_ul
            else:
                y_lr = Point((y[0], self.eval(y[0])))
                y_ul = Point(((y[1] - self.b) / self.a, y[1]))
                return Line((y_ul, y_lr))

        # commented - returns line segment
        # if self.eval(y[0]) < y[1]:
        # y_lr = Point((y[0],self.eval(y[0])))
        # y_ul = Point(( (y[1] -self.b)/self.a , y[1] ))
        # return Line((y_ul, y_lr))
        # else:
        # return None

    def contains_point(self, y, strictly=False):
        """returns True if the point y is contained in the line"""
        if strictly:
            if self.y1.is_close(y) or self.y2.is_close(y):
                return False
        else:
            if self.y1.is_close(y) or self.y2.is_close(y):
                return True

        if self.y1[0] > self.y2[0]:
            return Line.contains_point(Line((self[1], self[0])), y)
        assert self.y1[0] <= self.y2[0]
        if y[0] < self.y1[0] or self.y2[0] < y[0]:
            return False
        return np.isclose(y[1], self.eval(y[0]))

    def intersect(
        self,
        other: Line,
        self_segment=True,
        other_segment=True,
        self_direction_lr=False,
        self_direction_ul=False,
    ):
        """returns an intersecion point of self and other if it exists else None"""

        x = (self.b - other.b) / (other.a - self.a)
        if self_segment:
            if not (self.y1[0] <= x and x <= self.y2[0]):
                # print(f"bla1")
                return False
        if other_segment:
            if not (other.y1[0] <= x and x <= other.y2[0]):
                # print(f"{other.y1=}")
                # print(f"{other.y2=}")
                # print(f"{x=}")
                # print(f"bla2")
                return False
        if (
            self_direction_ul
        ):  # only return an intersection in the lower right direction
            # if self.y2[0] < x: return False
            if x < self.y2[0]:
                return False
            # pass
            # return False
        if self_direction_lr:  # only return an intersection in the upper left direction
            if x < self.y1[0]:
                return False
            # return False

        return Point((x, self.eval(x)))

    def __mul__(self, other):

        if isinstance(self, Line):
            round(other * 2)  # throw error if not number
            return Line((Point(-self.y1.val), Point(-self.y2.val)))

        else:
            raise NotImplementedError

    def __add__(self, other):

        if __debug__:
            pass
            # assert self.points[0][0] < self.points[1][0], 'line {other=} not sorted'
            # assert not(self.points[0] < self.points[1] or self.points[1] < self.points[0])

        if isinstance(other, Line):
            if __debug__:  # check that both lines are sorted and stable
                pass
                # assert other.points[0][0] <= other.points[1][0], 'line {self=} not sorted'
                # assert not(other.points[0] < other.points[1] or other.points[1] < other.points[0])

            return Line(
                (self.points[0] + other.points[0], self.points[1] + other.points[1])
            )

        elif isinstance(other, Point):
            return Line((self.points[0] + other, self.points[1] + other))

        else:
            print(f"__add__ not implemented for Line and {type(other)=}, {other=}")
            raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        return f"Line{self.points}"


@dataclass
class MSPInstances:
    preset: str = "all"
    options: dict = None
    filename_list: list[str] = None
    max_instances: int = 0
    m_options: tuple[int] = (2, 3, 4, 5)  # subproblems
    p_options: tuple[int] = (2, 3, 4, 5)  # dimension
    generation_options: tuple[str] = ("l", "m", "u", "ul")  # generation method
    ignore_ifonly_l: bool = False  # if true ignore MSP where method i only l
    size_options: tuple[int] = (50, 100, 150, 200, 300, 600)  # subproblems size
    seed_options: tuple[int] = (1, 2, 3, 4, 5)

    def instance_name_dict(problem_file):
        filename = problem_file
        problem_file = problem_file.split(".json")[0]
        problem_file, seed = problem_file.split("_")
        _, p, size, method, M = problem_file.split("-")
        size = size.split("|")[0]
        p, M, size, seed = int(p), int(M), int(size), int(seed)
        D = {
            "filename": filename,
            "p": p,
            "method": method,
            "M": M,
            "size": size,
            "seed": seed,
        }
        return D

    def instance_name_dict_keys(problem_file):
        D = MSPInstances.instance_name_dict(problem_file)
        return (D["M"], D["size"], D["p"], D["seed"])

    def __post_init__(self):
        all_problems = os.listdir("instances/problems/")
        # print(f"{all_problems=}")
        all_problems = sorted(all_problems, key=MSPInstances.instance_name_dict_keys)

        self.filename_list = []

        match self.preset:
            case "all":
                pass
            case "2d":
                self.p_options = (2,)
            case "algorithm1":
                self.generation_options = ["m", "u", "l"]  # generation method
                self.size_options = (50, 100, 150, 200, 300)  # subproblems size

            case "algorithm1_only_l":
                self.generation_options = ["l"]  # generation method
                self.size_options = (50, 100, 150, 200, 300)  # subproblems size

            case "algorithm1_largest":
                self.generation_options = ["m", "u", "l"]  # generation method
                self.size_options = (50, 100, 150, 200, 300)  # subproblems size
                self.p_options = (5,)
                self.m_options = (5,)

            case "grendel_test":
                self.filename_list = [
                    "prob-2-100|100-ll-2_1.json",
                    "prob-4-100|100-ll-2_1.json",
                    "prob-4-100|100|100-lll-3_1.json",
                    "prob-4-100|100|100-mmm-3_1.json",
                    "prob-5-100|100|100-mmm-3_1.json",
                    "prob-5-100|100|100|100|100-mmmmm-5_1.json",
                    # 'prob-4-200|200|200|200|200-lllll-5_5.json'
                ]
                self.max_instances = len(self.filename_list)
            case "algorithm2":
                self.generation_options = ["m", "u", "l"]  # generation method
                # self.p_options = (4,)
                # self.m_options = (4,)
                self.size_options = (50, 100, 150, 200, 300)  # subproblems size
            case "algorithm2_test":
                self.seed_options = (0,)  # ignora alle other test problems
                subsets_list = []
                subsets_list.append(
                    ("sp-2-10-u_1.json", "sp-2-10-u_1.json", "sp-2-10-u_2.json")
                )
                subsets_list.append(
                    ("sp-2-50-u_1.json", "sp-2-50-u_1.json", "sp-2-10-u_1.json")
                )
                subsets_list.append(
                    ("sp-2-100-u_1.json", "sp-2-100-l_1.json", "sp-2-100-u_1.json")
                )
                subsets_list.append(
                    ("sp-4-100-u_1.json", "sp-4-100-l_1.json", "sp-4-100-u_1.json")
                )
                subsets_list.append(
                    ("sp-4-100-u_2.json", "sp-4-100-l_1.json", "sp-4-100-u_2.json")
                )
                for subsets in subsets_list:
                    self.filename_list.append(MinkowskiSumProblem.from_subsets(subsets))
            case _:
                print(f"preset '{self.preset}' not recognised")
                raise NotImplementedError

        for filename in all_problems:
            instance_dict = MSPInstances.instance_name_dict(filename)

            if self.ignore_ifonly_l and set(instance_dict["method"]).issubset(
                set(("l",))
            ):
                continue
            if all(
                (
                    instance_dict["p"] in self.p_options,
                    instance_dict["M"] in self.m_options,
                    set(instance_dict["method"]).issubset(set(self.generation_options)),
                    instance_dict["size"] in self.size_options,
                    instance_dict["seed"] in self.seed_options,
                    (
                        self.preset != "algorithm1"
                        or (not (instance_dict["p"] == 5 and instance_dict["M"] == 5))
                    ),  # if algorithm 1 then not p=m=5
                )
            ):
                self.filename_list.append(filename)

        # limit number of files
        if self.max_instances:
            self.filename_list = self.filename_list[: self.max_instances]

    def filter_out_solved(self, save_prefix: str, solved_folder: str):
        self.not_solved = []
        self.solved = []
        for p in self.filename_list:
            filename = p if type(p) == str else p.filename
            if save_prefix + filename in os.listdir(solved_folder):
                self.solved.append(p)
            else:
                self.not_solved.append(p)

        print(f"|solved| = {len(self.solved)}    |not solved| = {len(self.not_solved)}")

        self.filename_list = self.not_solved

    def partition(self, n, k):
        """partitions the instance into n partitions and returns partition k"""
        self.filename_list = [
            file for i, file in enumerate(self.filename_list) if i % (n) == k
        ]

    def __repr__(self):
        return (
            f"TestInstances(size='{len(self.filename_list)}', preset='{self.preset}')"
        )

    def __iter__(self) -> iter[MinkowskiSumProblem]:
        return (
            (
                filename
                if isinstance(filename, MinkowskiSumProblem)
                else MinkowskiSumProblem.from_json("./instances/problems/" + filename)
            )
            for filename in self.filename_list
        )


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

    def __repr__(self):
        return f"{str(self.data)}"


class LinkedList:
    def __init__(self):
        self.head = None

    def __repr__(self):
        node = self.head
        nodes = []
        while node is not None:
            nodes.append(str(node.data))
            node = node.next
        nodes.append("None")
        return " -> ".join(nodes)

    def __iter__(self):
        node = self.head
        # while node is not None:
        while node is not None:
            yield node
            node = node.next

    def add_first(self, node):
        node.next = self.head
        self.head = node
        self.prev = None

    def add_after(self, target_node_data, new_node):
        if self.head is None:
            raise Exception("List is empty")

        for node in self:
            if node.data == target_node_data:
                new_node.next = node.next
                node.next = new_node
                return

        raise Exception("Node with data '%s' not found" % target_node_data)

    def add_before(self, target_node_data, new_node):
        if self.head is None:
            raise Exception("List is empty")

        if self.head.data == target_node_data:
            return self.add_first(new_node)

        prev_node = self.head
        for node in self:
            if node.data == target_node_data:
                prev_node.next = new_node
                new_node.next = node
                return
            prev_node = node

        raise Exception("Node with data '%s' not found" % target_node_data)

    def remove_node(self, target_node_data):
        if self.head is None:
            raise Exception("List is empty")

        if self.head.data == target_node_data:
            self.head = self.head.next
            return

        previous_node = self.head
        for node in self:
            if node.data == target_node_data:
                previous_node.next = node.next
                return
            previous_node = node

        raise Exception("Node with data '%s' not found" % target_node_data)


@dataclass
class KD_Node:
    y: Point
    l: int
    parent: KD_Node = None
    LEFT: KD_Node = None
    RIGHT: KD_Node = None
    UB: Point = None
    LB: Point = None

    # def __repr__(self):
    # return f"{str(self.y)}"
    def __str__(self, level=0):
        ret = "\t" * level + repr(self.y) + f"l={self.l}" + "\n"

        if self.LEFT != None:
            ret += self.LEFT.__str__(level + 1)
        else:
            ret += "\t" * (level + 1) + "Ø \n"

        if self.RIGHT != None:
            ret += self.RIGHT.__str__(level + 1)
        else:
            ret += "\t" * (level + 1) + "Ø \n"

        return ret

    def __repr__(self):
        # return str(self.y)
        return f"KD_NODE(y={self.y}, parent={self.parent.y if self.parent else 'Ø'}, LEFT={self.LEFT.y if self.LEFT else 'Ø'}, RIGHT={self.RIGHT.y if self.RIGHT else 'Ø'}, UB = {self.UB}, LB = {self.LB})"


@dataclass
class KD_tree:
    def dominates_point_recursion(r: KD_Node, p: Point):
        # seperated for timing purposes

        if r.y <= p:
            return True
        if r.LEFT != None and p > r.LEFT.LB:
            return KD_tree.dominates_point_recursion(r.LEFT, p)
        if r.RIGHT != None and p > r.RIGHT.LB:
            return KD_tree.dominates_point_recursion(r.RIGHT, p)
        return False

    def dominates_point(r: KD_Node, p: Point):
        """checks if point is dominated by the KD-tree rooted at r

        Args:
            p (Point): point

        Returns:
            1, if p is dominated by a point in the KD-tree rooted at r,
            0, otherwise

        """
        return KD_tree.dominates_point_recursion(r, p)

    def get_UB(r: KD_Node, p: Point):
        return Point(np.maximum(r.UB.val, p.val))
        # old
        # return Point([max(r.UB[i], p[i]) for i in range(p.dim)])

    def get_LB(r: KD_node, l: int, p: Point):
        return Point(np.minimum(r.LB.val, p.val))
        # return Point([min(r.LB[i], p[i]) for i in range(p.dim)])

    def insert_recursion(r: KD_Node, l: int, p: Point):
        # seperated for timing purposes
        # update r.UB, r.LB
        r.UB = KD_tree.get_UB(r, p)
        r.LB = KD_tree.get_LB(r, l, p)

        # compare l-th component of p and r
        # print(f"{r,l,p =}")
        if p[l] < r.y[l]:
            if r.LEFT == None:
                r.LEFT = KD_Node(p, (l + 1) % p.dim, r, UB=p, LB=p)
            elif r.LEFT != None:
                KD_tree.insert_recursion(r.LEFT, (l + 1) % p.dim, p)
        elif p[l] > r.y[l]:
            if r.RIGHT == None:
                r.RIGHT = KD_Node(p, (l + 1) % p.dim, r, UB=p, LB=p)
            elif r.RIGHT != None:
                KD_tree.insert_recursion(r.RIGHT, (l + 1) % p.dim, p)

    def insert(r: KD_Node, l: int, p: Point):
        return KD_tree.insert_recursion(r, l, p)
