"""
methods/algorithms for filtering out nondominated points
specifically for a single set (Non minkowski sum problem)

IMPLEMENTED methods:
    basic_filter(Y) -> Yn
    naive_filter(Y) -> Yn
    lex_sort(Y) -> sorted(Y)
    unidirectional_filter(Y) -> Yn

    MS_sum(Y_list) -> Y_ms, Minkowski sum of sets in Y_list
    MS_naive_filter(Y_list) -> N(Y_ms)
    MS_sequential_filter(Y_list) -> N(Y_ms)
    MS_doubling_filter(Y_list) -> N(Y_ms)
"""

from src.asmo.classes.pointsets import Point, PointList, Line
import numpy as np
import os
import subprocess  # for running c execute
from collections import deque  # for fast leftappend
from operator import itemgetter  # for lexsort function to define keys
from typing import Iterable
import math
import uuid
import random


def basic_filter(Y: PointList) -> PointList:
    """
    input: PointList
    output: PointList with all nondominated points removed

    ALG:
        For each point y, check if any other point y2 dominates y: if no, add y to Yn
    """
    Yn = []

    for y in Y:
        for y2 in Y:
            if y2 < y:
                break
        else:
            Yn.append(y)

    return PointList(Yn)


def naive_filter(Y: PointList, MCtF=False) -> PointList:
    """
    input: PointList
    output: PointList with all nondominated points removed

    ALG:
        For each point y, check if any other point y2 dominates y: if no, add y to Yn
    """
    Yn = deque()

    for i, y in enumerate(Y):
        dominated_indices_y = set()
        for j, y2 in enumerate(Yn):
            if y2 <= y:
                break  # discard y
            if y2 > y:
                dominated_indices_y.add(j)  # record dominance
        else:
            # remove dominated points
            Yn = deque(
                (y_ for j_, y_ in enumerate(Yn) if j_ not in dominated_indices_y)
            )
            # add nondominated candidate
            if MCtF and dominated_indices_y:
                Yn.appendleft(y)
            else:
                Yn.append(y)

    # Y.points = Yn
    return PointList(Yn)


def two_phase_filter(Y: PointList) -> PointList:
    """Two phase filter from Chen2012 for filtering a list of points

    Args:
        Y (PointList): Yn (PointList) set of nondominated points

    Returns: PointList with all nondominated points removed
    """

    Y = lex_sort(Y)

    Yn = PointList(())
    Yn.dim = Y.dim
    Yn.points = list(Yn.points)
    # phase 1
    for i, y in enumerate(Y):
        if not Yn.weakly_dominates_point(y):
            Yn.points.append(y)
    # print(f"Phase 1 result: {Yn=}")
    # phase 2
    Yn_new = PointList()
    Yn_new.dim = Yn.dim
    Yn_new.points = list(Yn_new.points)
    for i in range(len(Yn)):
        y = Yn[-i + 1]
        if not Yn_new.weakly_dominates_point(y):
            Yn_new.points.append(y)
    return Yn_new


def KD_filter(Y: PointList) -> PointList:
    """Kd-tree filtering algorithm from Chen2012

    Args:
        Y (PointList)

    Returns:
        Yn (PointList) set of nondominated points

    Returns: PointList with all nondominated points removed

    """
    Y = lex_sort(Y)  # sort input list
    R = []
    for i, y in enumerate(Y):
        if i == 0:
            r = KD_Node(Y[0], 0)
            r.UB = r.y
            r.LB = r.y
            k = 0
            R.append(y)
            continue

        if not KD_tree.dominates_point(r, y):
            # print(f"{y=}")
            KD_tree.insert(r, 0, y)
            R.append(y)
            k += 1

    r = None
    Yn = []
    for i, y in enumerate(reversed(R)):
        if i == 0:
            r = KD_Node(y, 0)
            r.UB = r.y
            r.LB = r.y
            Yn.append(y)
            continue

        if not KD_tree.dominates_point(r, y):
            KD_tree.insert(r, 0, y)
            Yn.append(y)

    Yn = PointList(Yn)

    return Yn


def lex_sort(Y: PointList) -> PointList:
    """
    input: PointList
    output: lexicographically sorted PointList Y

    source https://stackoverflow.com/questions/38277143/sort-2d-numpy-array-lexicographically
    """
    if len(Y) == 0:
        return Y
    Y.points = sorted(Y.points, key=itemgetter(*range(Y.dim)))

    for i in range(len(Y.points) - 1):  # simple but not exhaustive correctness check
        assert not Y[i] > Y[i + 1], f"{Y[i]=},{Y[i+1]=} "

    return PointList(Y.points)


def unidirectional_filter(Y: PointList, duplicates_allowed=False) -> PointList:
    """
    input: PointList, bool - allowed_duplicates
    output: PointList with all nondominated points removed
    """

    Y = lex_sort(Y)

    # p = 2
    assert Y[0].val.shape[0] <= 2, "dim p > 2 NOT IMPLEMENTED"
    Yn = []

    if duplicates_allowed:
        for y in Y:
            if Yn == [] or not Yn[-1] < y:
                Yn.append(y)
                # assert not PointList(Yn).dominates_point(y), f"{Yn=}, {y=}"

    else:  # if duplicates not allowed
        for y in Y:
            if Yn == [] or not Yn[-1] <= y:
                Yn.append(y)
    return PointList(Yn)


def call_c_nondomDC(call_id: str, max_time=None, logger=None):
    # current_d = os.getcwd()
    # move to c folder and execute
    # os.chdir('/Users/au618299/Desktop/cythonTest/nondom/')
    # subprocess.call(['./nondom',call_id])
    # return to initial directory
    # os.chdir(current_d)
    # subprocess.call(['/Users/au618299/Desktop/cythonTest/nondom/./nondom',call_id])
    assert "nondom" in os.listdir(), os.listdir()
    # subprocess.call(['./nondom',call_id])
    p = subprocess.Popen(["./nondom", call_id])

    try:
        if max_time:
            p.wait(timeout=max_time * 60)
        else:
            p.wait()
    except subprocess.TimeoutExpired:
        print(f"Process timed out after {max_time} seconds")
        p.kill()
        print("Process killed due to timeout")
        if logger:
            logger.warning("Process timed out after {max_time} seconds {call_id=}")


def call_c_ND_pointsSum2(
    call_id: str, max_time=None, max_gb=None, logger=None, verbose=False
):
    assert "ND_pointsSum2" in os.listdir(), os.listdir()

    # print(f"calling subprocess ")
    if max_gb:
        if verbose:
            p = subprocess.Popen(["./ND_pointsSum2", call_id, str(max_gb)])
        else:
            p = subprocess.Popen(
                ["./ND_pointsSum2", call_id, str(max_gb)], stdout=subprocess.DEVNULL
            )
    else:
        if verbose:
            p = subprocess.Popen(["./ND_pointsSum2", call_id])
        else:
            p = subprocess.Popen(
                ["./ND_pointsSum2", call_id], stdout=subprocess.DEVNULL
            )

    try:
        if max_time:
            p.wait(timeout=max_time * 60)
        else:
            p.wait()
    except subprocess.TimeoutExpired:
        print(f"Process timed out after {max_time} seconds")
        p.kill()
        print("Process killed due to timeout")
        if logger:
            logger.warning("Process timed out after {max_time} seconds {call_id=}")
            logger.info(f"{p.returncode=}")
        print(f"{p.returncode=}")


def nondomDC_wrapper(Y: PointList) -> PointList:
    # A python wrapper for the c implementation of NonDomDC [Bruno Lang]
    call_id = str(uuid.uuid4())
    # out_file = fr"/Users/au618299/Desktop/cythonTest/nondom/temp/pointsIn-{call_id}" # c script directory
    out_file = rf"temp/pointsIn-{call_id}"  # c script directory
    Y.save_raw(out_file)
    assert os.path.exists(out_file), f"{out_file=}"

    try:
        out_str = call_c_nondomDC(call_id)
    finally:
        os.remove(out_file)
    # in_file = filepath = fr"/Users/au618299/Desktop/cythonTest/nondom/temp/pointsOut-{call_id}" # c script directory
    in_file = filepath = rf"temp/pointsOut-{call_id}"  # c script directory
    assert os.path.exists(in_file), f"{in_file,out_str=}"
    Yn = PointList.from_raw(in_file)
    # print(f"{in_file=}")
    try:
        Yn = PointList.from_raw(in_file)
    except FileNotFoundError:
        print(f"File not found")
        print(f"{in_file=}")
        return None

    if True:  # clear temp folder
        # os.remove(out_file)
        os.remove(in_file)
    return Yn.removed_duplicates()


def ND_pointsSum2_wrapper(A: PointList, B: PointList) -> PointList:
    # A python wrapper for the c implementation of ND_pointsSum2 [Bruno Lang]
    call_id = str(uuid.uuid4())
    # out_file = fr"/Users/au618299/Desktop/cythonTest/nondom/temp/pointsIn-{call_id}" # c script directory
    out_fileA = rf"temp/pointsInA-{call_id}"  # c script directory
    out_fileB = rf"temp/pointsInB-{call_id}"  # c script directory
    A.save_raw(out_fileA)
    B.save_raw(out_fileB)

    try:
        call_c_ND_pointsSum2(call_id)
    finally:
        os.remove(out_fileA)
        os.remove(out_fileB)

    in_file = filepath = rf"temp/pointsOut-{call_id}"  # c script directory
    try:
        Yn = PointList.from_raw(in_file)
    except FileNotFoundError:
        return None

    if True:  # clear temp folder
        os.remove(in_file)
    return Yn.removed_duplicates()


def N(Y=PointList, **kwargs) -> PointList:
    """'best' implemented nondominance filter"""
    if Y[0].dim <= 2:
        return unidirectional_filter(Y, *kwargs)
    else:
        return nondomDC_wrapper(Y)


def MS_sum(Y_list=list[PointList], operator="+") -> PointList:
    """
    input: list of PointList
    output: Minkowski sum of sets
    """
    assert operator in ("+", "-", "*")

    Y_ms = Y_list[0]
    for s in range(1, len(Y_list)):
        Y_ms_new = []
        Y_s = Y_list[s]
        for y_ms in Y_ms:
            for y_s in Y_s:
                if operator == "+":
                    Y_ms_new.append(y_ms + y_s)
                if operator == "-":
                    Y_ms_new.append(y_ms - y_s)
                if operator == "*":
                    Y_ms_new.append(y_ms * y_s)
        Y_ms = Y_ms_new

    return PointList(Y_ms)


def MS_naive_filter(Y_list=list[PointList]) -> PointList:
    """
    input: list of PointList
    output: nondominated points of Minkowski sum of sets Y_list
    """
    Y = MS_sum(Y_list)
    Yn = N(Y)

    return PointList(Yn)


def MS_sequential_filter(Y_list=list[PointList], N=N) -> PointList:
    """
    input: list of PointList
    output: nondominated points of Minkowski sum of sets Y_list
    """
    Y_ms = N(Y_list[0])

    for s in range(1, len(Y_list)):
        # print(f"{s=}")
        # print(f"{len(Y_ms)=}")
        # Y_ms = N(Y_ms + N(Y_list[s]))
        # Y_ms = ND_pointsSum2_wrapper(Y_ms, N(Y_list[s])) TODO: NEWEST
        Y_ms = N(Y_ms + N(Y_list[s]))
        Y_ms = Y_ms.removed_duplicates()
        # assert Y_ms == N(Y_ms), f"{len(Y_ms),len(N(Y_ms)),len(Y_ms.removed_duplicates())=}"
        if Y_ms is None:
            return None

    return PointList(Y_ms)


def MS_doubling_filter(
    Y_list=list[PointList], MS_filter_alg=MS_sequential_filter
) -> PointList:
    """
    input: list of PointList
    output: nondominated points of Minkowski sum of sets Y_list
    """

    s = len(Y_list)
    S = Y_list
    while s > 1:
        S_new = []
        for k in range(math.floor(s / 2)):
            S_new.append(MS_filter_alg((S[2 * k], S[2 * k + 1])))
        if s % 2 != 0:
            S_new.append(S[-1])
        s = math.ceil(s / 2)
        S = S_new
    return S[0]


def lex_sort_linked(Y: PointList) -> PointList:
    """function for sorting p = 2$"""
    assert Y.dim <= 2, "dim p > 2 NOT IMPLEMENTED"

    llist = LinkedList()
    llist.add_first(Node(Y[0]))
    llist.head.prev = None

    for y_current in Y[1:]:
        for N in llist:
            if y_current.val[0] >= N.data.val[0]:
                continue
            # traverse linked list until y lex dominated the node N
            if y_current.lex_le(N.data):
                new_node = Node(y_current)
                # add before N
                llist.add_before(N.data, new_node)

                # remove N.data and all dominated children
                first_nondom = N
                # while first_nondom != None and y_current < first_nondom.data:
                # first_nondom = first_nondom.next
                # print(f"removing {first_nondom}")
                new_node.next = first_nondom
                break
            #  elif N.data < y_current:
            #      break
            prev = N
        else:
            if N.next == None:
                new_node = Node(y_current)
                N.next = new_node
    return PointList((N.data for N in llist.__iter__()))


def lex_filter(Y: PointList) -> PointList:
    """function for filtering out dominated points using linked lists for p = 2$"""
    assert Y.dim <= 2, "dim p > 2 NOT IMPLEMENTED"

    llist = LinkedList()
    llist.add_first(Node(Y[0]))
    llist.head.prev = None

    for y_current in Y[1:]:
        for N in llist:
            if y_current.val[0] >= N.data.val[0]:
                if y_current.val[1] >= N.data.val[1]:
                    break
                else:
                    continue
            # traverse linked list until y lex dominated the node N
            if y_current.lex_le(N.data):
                new_node = Node(y_current)
                # add before N
                llist.add_before(N.data, new_node)

                # remove N.data and all dominated children
                first_nondom = N
                while first_nondom != None and y_current < first_nondom.data:
                    first_nondom = first_nondom.next
                    # print(f"removing {first_nondom}")
                new_node.next = first_nondom
                break
            #  elif N.data < y_current:
            #      break
            prev = N
        else:
            if N.next == None:
                if not N.data < y_current:
                    new_node = Node(y_current)
                    N.next = new_node
    return PointList((N.data for N in llist.__iter__()))


def induced_UB(Y: PointList, line=False, assumption="consecutive") -> PointList:
    """Induced upper bound set from pointlist Y, points are assumed to be consecutive in Yn"""
    # arg assumption in [consecutive, supported, nonconsecutive]
    assert assumption in ["consecutive", "supported", "nonconsecutive", "localNadir"]

    Y = N(Y)
    Y = lex_sort(Y)
    U = []
    seen = set()  # for spotting duplicates
    if line:
        U.append(Y[0])
        for i in range(len(Y) - 1):
            if assumption == "consecutive":
                u = Point((Y[i + 1][0], Y[i][1]))
            elif assumption in {"nonconsecutive", "localNadir"}:
                u = Point((Y[i][0], Y[i + 1][1]))
            if assumption != "supported":
                U.append(u)
            U.append(Y[i + 1])
    else:
        for i in range(len(Y) - 1):
            if Y[i + 1] not in seen:  # ignore duplicates
                seen.add(Y[i + 1])
                u = Point((Y[i + 1][0], Y[i][1]))
                U.append(u)
    if assumption == "localNadir":
        U = PointList([Y[0]] + U + [Y[-1]])
        assert U.dim == 2
    else:
        U = PointList(U)
    return U


def find_generator_U(Y1: PointList, Y2: PointList) -> PointList:
    """
    input: two sets Y1, Y2, where Y1 contains (global) lex min solutions.
    output: A set of generator upper bound points Uc as PointList
    """

    def get_i(points: PointList, q: Point):
        """
        intervals: a PointList y1_1 < y2_1 < y3_1 ...
        q: a Point
        returns the id i of PointList where yi_1 == q_1
        """
        if points[0][0] == q[0]:
            return 0
        if points[-1][0] <= q[0]:
            return -1

        # assert that Q is sorted (consequence of Y2 sorted)
        for i, y in enumerate(points):
            if points[i][0] <= q[0] and q[0] < points[i + 1][0]:
                return i

    Y1 = N(Y1)
    Y2 = N(Y2)

    Y = N(Y1 + Y2)

    y_ul = Y1[0]
    y_lr = Y1[-1]

    u_current = y_ul
    Uc = [u_current]
    Q = PointList((u_current,)) + Y2

    while u_current != y_lr:
        # assert Q == PointList((u_current,)) + Y2
        # determine right movement
        Q_bar = [q for q in Q if Y[get_i(Y, q)][1] == q[1]]
        l1 = max([Y[get_i(Y, q) + 1][0] - q[0] for q in Q_bar])

        # determine down movement
        Q = Point((l1, 0)) + Q
        u_current = u_current + Point((l1, 0))
        l2 = min([q[1] - Y[get_i(Y, q)][1] for q in Q])

        # Update Q, u_current and Uc
        Q = Point((0, -l2)) + Q
        u_current = u_current + Point((0, -l2))
        Uc.append(u_current)

    return PointList(Uc)


def U_dominates_L(U: PointList, L: PointList) -> bool:
    """
    Checks if the lower bound L is dominated by the upper bound U
    input :
        U : list of non-dominated points as tuples
        L : list of supported points which make up the lower bound set
    """
    assert all((L.dim == U.dim, L.dim == 2))
    # y = sorted(set(L)) # sort non-dominated points

    if L.is_complete:  # Assume Yn = Ln, ie LB not defined by the convex hull
        return N(U) < N(L)

    L = lex_sort(N(L))
    local_nadir_points = induced_UB(U)

    if len(L) == 1:
        return U.dominates_point(L[0])

    # Check that all extreme points of L are dominated
    for l in L:
        for u in U:
            if u < l:
                break
        else:  # finally, if loop terminates normally
            # print(f"The LB point {l=} is not dominated by any point of U")
            return False

    # Check that all line segments of L are dominated
    for i in range(len(set(L)) - 1):
        # define linear function (line between l[i] and l[i+1])
        lin_fct = lambda x: L[i][1] + (L[i + 1][1] - L[i][1]) / (
            L[i + 1][0] - L[i][0]
        ) * (x - L[i][0])
        for n in local_nadir_points:
            if L[i][0] <= n[0] and n[0] <= L[i + 1][0]:
                if n[1] > lin_fct(n[0]):
                    # print(f"line between {L[i],L[i+1]} is not dominated by nadir-point {n}")
                    return False
                if math.isclose(n[1], lin_fct(n[0])):
                    return False
    # if loop ends, the node is not dominated by the upper bound set
    return True


def get_partial(Y, level="all", seed=0):
    Y = N(Y)
    Y2e_points = [y for y in Y if y.cls == "se"]
    Y2other_points = [y for y in Y if y.cls != "se"]
    random.seed(seed)
    random.shuffle(Y2other_points)
    match level:
        case "all":
            return Y
        case "lexmin":
            return PointList((Y[0], Y[-1]))
        case "extreme":
            return PointList(Y2e_points)
        # case float():
        case _:
            to_index = math.floor(float(level) * len(Y2other_points))
            return PointList(Y2e_points + Y2other_points[:to_index])


def get_lex_min(L):

    for i, y in enumerate(L.iter_endpoints()):
        if i == 0:
            y_lr, y_ul = y
        else:
            if y[0] <= y_ul[0] and (not y_ul < y):
                y_ul = y
            if y[1] <= y_lr[1] and (not y_lr < y):
                y_lr = y


def N_lines(L: Iterable[Line], add_horisontal="right"):

    if add_horisontal == "right":
        assert L[-1].a == 0, L[-1].a  # add horizontal line as last segment
    else:
        assert L[0].a == 0, L[0].a  # add horizontal line as last segment

    Y = PointList(L.iter_endpoints())
    y_ideal, y_nadir = Y.get_ideal(), Y.get_nadir()
    inf = np.linalg.norm(y_ideal.val - y_nadir.val)
    inf = inf.round() + 1

    # touching points

    # L.plot()
    intersection_points = []

    for i, l1 in enumerate(L):
        for j, l2 in enumerate(L):
            if j >= i:
                continue
            # assume j < i
            # print(f"{i,j=}")
            if y_intersection := l1.intersect(l2):
                # if any(l.contains_point(y_intersection, strictly= True) for l in self): continue
                # y_intersection.plot(ax=ax, s=60, color = 'black', marker = 'x')
                y_intersection.lines = [l1, l2]
                if not any(
                    int_point.is_close(y_intersection)
                    for int_point in intersection_points
                ):
                    intersection_points.append(y_intersection)

    intersection_points = [
        y for y in intersection_points if not any(y2.dominates_point(y) for y2 in L)
    ]  # remove dominated endpoints
    # intersection_points = [y for y in intersection_points if not any(y2.contains_point(y, strictly = True) for y2 in self if y2 != y.line)] # remove dominated endpoints
    intersection_points = lex_sort(PointList(intersection_points))
    # intersection_points.plot(ax=ax, s=60, color = 'black', marker = 'x')
    # add all endpoints
    all_end_points = []
    for l in L:
        y1 = l[0]
        y2 = l[1]
        y1.line, y2.line = l, l
        if y1 not in all_end_points:
            y1.lines = [l]
            all_end_points.append(y1)
        else:
            all_end_points[all_end_points.index(y1)].lines.append(l)
        if y2 not in all_end_points:
            y2.lines = [l]
            all_end_points.append(y2)
        else:
            all_end_points[all_end_points.index(y2)].lines.append(l)

        # all_end_points += [y1, y2]
        # y1.plot(ax=ax, s=60, color = 'black', marker = 'x')
        # y2.plot(ax=ax, s=60, color = 'black', marker = 'x')

    ND_points = [
        y for y in all_end_points if not any(y2.dominates_point(y) for y2 in L)
    ]  # remove dominated endpoints
    ND_points = [
        y
        for y in ND_points
        if not any(y2.contains_point(y, strictly=True) for y2 in L if y2 != y.line)
    ]  # remove dominated endpoints
    ND_points = PointList(ND_points)
    print(f"{ND_points=}")
    # ND_points.plot(s=60, color = 'yellow', marker = 'x')

    print(f"{intersection_points=}")

    # all points
    all_points = lex_filter(PointList(list(ND_points) + list(intersection_points)))
    # all_points.plot(point_labels=True)
    new_lines = []
    # plt.show()
    for y1, y2 in zip(all_points, all_points[1::]):
        if y1.is_close(y2):
            continue
        # if y1.is_close(y2): continue
        # if y1 and y2 are endpoints the same line then add that line
        print(f"{y1,y2=}")
        print(f"{y1.lines=}")
        print(f"{y2.lines=}")
        if set(y1.lines).intersection(y2.lines) != set():
            l = Line((y1, y2))

        # else create a line from endpoint y1 along the steepest sloping line until the x-coord of y2 is reached
        else:
            # steepest_sloping_line = min((l_right for l_right in y1.lines if  y1[0] + 0.0001< l_right[1][0]), key = lambda line: line.a)
            steepest_sloping_line = min(
                (
                    l_right
                    for l_right in L
                    if l_right.contains_point(y1) and y1[0] + 0.0001 < l_right[1][0]
                ),
                key=lambda line: line.a,
            )
            print(f"{y1,y1.lines=}")
            print(f"{steepest_sloping_line=}")
            right_end_point = Point((y2[0], steepest_sloping_line.eval(y2[0])))
            print(f"{y1,y2=}")
            assert (
                right_end_point[1] >= y2[1] - 0.00001
            )  # we require all horizontal lines ie. the piecewise linear function that the input represents must be defined in the entire domain

            if y1.is_close(right_end_point):
                continue
            l = Line((y1, right_end_point))

        new_lines.append(l)

    # add horizontal line
    print(f"{all_points[-1]=}")

    # y_ideal, y_nadir = all_points.get_ideal(), all_points.get_nadir()
    # inf = np.linalg.norm(y_ideal.val - y_nadir.val)
    # inf = inf.round() + 1

    if add_horisontal == "right":
        y_lr = Point((all_points[-1][0] + inf, all_points[-1][1]))
        print(f"{y_lr=}")
        l = Line((all_points[-1], y_lr))
        print(f"{l=}")
        new_lines.append(l)
    elif add_horisontal == "left":
        y_ul = Point((all_points[0][0] - inf, all_points[0][1]))
        l = Line((y_ul, all_points[-1]))
        print(f"{l=}")
        new_lines.append(l)

    else:
        pass

    # for l in new_lines:
    # l.plot(ax=ax, color = 'yellow')
    # plt.show()
    return PointList(new_lines)


def lower_bound_from_supported(Y: PointList):
    """Returns line representation of a set of supported points

    Args:
        Y (PointList): A set of points - generated by the phase one methods

    Returns: A PointList containing Line segments representing the lower bound - excluding any vertical segments.
    """
    Y = lex_sort(Y)  # sort the incoming list to identify the lexmin vectors
    # add a point to the lower right to create a horizontal line from the lexmin y_lr to infininy
    Y = PointList(
        list(Y.points)
        + [Point((Y[-1][0] + Line((Y[0], Y[-1])).length() + 1, Y[-1][1]))]
    )

    L = PointList([Line((y1, y2)) for y1, y2 in zip(Y, Y[1::])])

    return L


def update_lower_bound(L: PointList, y: Point) -> PointList:
    """update lower bound from right

    Args:
        L (PointList): a stable line set lower bound
        L (y): a nd point

    Returns: PointList

    """

    Ldom = None
    L_new = []

    if any((l <= y for l in L.iter_endpoints())):
        return L

    # determine relevant search area - ie. line which dominates y
    # if y not in L.iter_endpoints():
    elif not any((l.is_close(y) for l in L.iter_endpoints())):
        Ldom = [l for l in L if l.dominates_point(y)]
        if __debug__:
            for l1 in Ldom:
                for l2 in Ldom:
                    assert set(l1.points).intersection(set(l2.points)) != set(), str(
                        (l1, l2)
                    )
            # print(f"{Ldom=}")
        Ldom = min([l for l in Ldom], key=lambda x: x[0][0])

        y_left = Ldom.dominates_point(y)
        y_right = Point((Ldom[1][0], y_left[1]))

        new_line_1 = Line((Ldom[0], y_left))
        new_line_2 = Line((y_left, y_right))

        # new_line_1.plot(ax=ax, l = 'new_line_1', linewidth = 3.5)
        # new_line_2.plot(ax=ax, l = 'new_line_2', linewidth = 3.5)

        L_new = [new_line_1, new_line_2]
    else:
        pass

    for l in L:
        if l == Ldom:
            continue

        # 'close' non-empty search areas to the right of the point y
        if y[0] <= l[0][0] and l.a != 0:
            l_horizontal = Line((l[0], Point((l[1][0], l[0][1]))))
            L_new.append(l_horizontal)
        else:
            L_new.append(l)

    L = PointList(L_new)  # determine relevant search area - ie. line which dominates y

    return L
