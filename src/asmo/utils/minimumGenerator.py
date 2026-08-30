from src.asmo.classes.pointsets import PointList
import src.asmo.utils.mspMethods as methods
import math
import matplotlib.pyplot as plt
import pyomo.environ as pyomo
import numpy as np
from scipy.spatial import KDTree

DEFAULT_SOLVER = "cplex_direct"


def build_model(Y_list) -> pyomo.ConcreteModel():
    print(f"Setting up model..")
    P = Y_list[0][0].dim
    assert all((P == Y.dim for Y in Y_list))

    Yn = methods.MS_sequential_filter(Y_list)

    S = list(range(len(Y_list)))
    J = list(range(len(Yn)))
    I_dict = {s: list(range(len(Y_list[s]))) for s in S}

    # Define your model
    model = pyomo.ConcreteModel()

    # Define your sets
    model.S = pyomo.Set(initialize=S)
    model.J = pyomo.Set(initialize=J)
    model.I_s = pyomo.Set(model.S, initialize=I_dict)

    # Create an intermediate set
    def IS_rule(model):
        return [(s, i) for s in S for i in model.I_s[s]]

    model.IS = pyomo.Set(within=model.S * model.J, initialize=IS_rule)

    # Define binary variables
    model.x = pyomo.Var(model.IS, within=pyomo.Binary)
    model.w = pyomo.Var(model.IS, model.J, within=pyomo.Binary)

    # Define constraints
    model.cons = pyomo.ConstraintList()
    for s in S:  # at least one solution to each subproblem
        model.cons.add(sum(model.x[s, i] for i in model.I_s[s]) >= 1)
        # at most one part of j can be composed of points in subproblem s
        for j in model.J:
            model.cons.add(sum(model.w[s, i, j] for i in model.I_s[s]) == 1)

    for s, i in model.IS:
        for j in model.J:
            # ys_i covers j only if ys_i is chosen
            model.cons.add(model.w[s, i, j] <= model.x[s, i])

    for j in model.J:
        for p in range(P):

            model.cons.add(
                sum(model.w[s, i, j] * Y_list[s][i][p] for s in S for i in I_dict[s])
                - Yn[j][p]
                == 0
            )

    # Define objective
    def objective_rule(model):
        return sum(model.x[s, i] for s, i in model.IS)

    model.obj = pyomo.Objective(rule=objective_rule, sense=pyomo.minimize)

    return model


def build_model_covering(
    Y_list, Yn, Yn_nongenerated, C_dict, Y_fixed, Y_reduced
) -> pyomo.ConcreteModel():
    print(f"Setting up covering model..")
    P = Y_list[0][0].dim
    S = len(Y_list)
    assert all((P == Y.dim for Y in Y_list))

    # Yn = methods.MS_sequential_filter(Y_list)

    # index sets
    S = list(range(len(Y_list)))
    J = list(range(len(Yn_nongenerated)))
    C = list(range(max([len(C_dict[Yn_nongenerated[j]]) for j in J])))
    I = list(range(max([len(Y) for Y in Y_list])))
    # print(f"{len(C)=}")
    # print(f"{S=}")
    # print(f"{J=}")
    # print(f"{I=}")

    Y_unfixed_dict = {s: tuple(set(Y_reduced[s]) - set(Y_fixed[s])) for s in S}
    Y_unfixed_set_dict = {s: set(Y_reduced[s]) - set(Y_fixed[s]) for s in S}
    # print(f"{[len(Y) for Y in Y_unfixed_dict.values()]=}")

    # print(f"{len(Yn),len(Yn_nongenerated)=}")

    # X_dict = {s : list(range(len(Y_list[s]))) for s in S}
    # X_dict = {s : list(range(len(Y_list[s]))) for s in S}
    # print(f"{Y_fixed=}")
    # print(f"{Y_reduced=}")
    # print(f"{Y_unfixed_dict=}")
    Yn_comb_dict = {j: C_dict[Yn_nongenerated[j]] for j in J}
    for k, v in Yn_comb_dict.items():
        # print(f"{Yn_comb_dict=}")
        assert len(v) > 1
    Yn_comb_reduced_dict = {
        j: [
            tuple(
                [ci if ci in Y_unfixed_set_dict[s] else None for s, ci in enumerate(c)]
            )
            for c in C_dict[Yn_nongenerated[j]]
        ]
        for j in J
    }

    Yn_comb_index = {j: tuple(range(len(C_dict[Yn_nongenerated[j]]))) for j in J}
    # print(f"{Yn_comb_dict[0]=}")
    # print(f"{Yn_comb_reduced_dict[0]=}")
    # print(f"{Yn_comb_index[0]=}")
    # Define your model
    model = pyomo.ConcreteModel()

    # Define your sets
    model.S = pyomo.Set(initialize=S)
    model.J = pyomo.Set(initialize=J)
    model.C = pyomo.Set(initialize=C)
    model.I = pyomo.Set(initialize=I)
    model.I_s = pyomo.Set(model.S, initialize=Y_unfixed_dict)
    model.J_c = pyomo.Set(model.J, initialize=Yn_comb_index)

    # Create an intermediate set of valid combinations
    def SI_rule(model):
        return [(s, i) for s in S for i in model.I_s[s]]

    def JC_rule(model):
        return [(j, c) for j in J for c in model.J_c[j]]

    model.SI = pyomo.Set(within=model.S * model.I, initialize=SI_rule)
    model.JC = pyomo.Set(within=model.J * model.C, initialize=JC_rule)

    # Define binary variables
    model.x = pyomo.Var(model.SI, within=pyomo.Binary)
    model.z = pyomo.Var(model.JC, within=pyomo.Binary)

    # Define constraints
    model.cons = pyomo.ConstraintList()

    for j in model.J:
        model.cons.add(sum(model.z[j, c] for c in Yn_comb_index[j]) >= 1)
    for j in model.J:
        for c in Yn_comb_index[j]:
            assert (j, c) in model.JC
            incident_s_ys = [
                (s, ys)
                for (s, ys) in enumerate(Yn_comb_reduced_dict[j][c])
                if ys is not None
            ]
            # print(f"{j,c,incident_s_ys=}")
            # print(f"{len(incident_s_ys)=}")
            assert len(incident_s_ys) > 0
            model.cons.add(
                model.z[j, c] * len(incident_s_ys)
                <= sum(model.x[s, ys] for s, ys in incident_s_ys)
            )

    #     if True:
    # for (s,i) in model.SI:
    # model.cons.add(model.x[s,i] >=1)

    if False:  # Check uniqueness
        # for each problem only one of the decision variables was chosen

        # for checking prob-2-200|200|200|200-mmmm-4_2.json
        # model.cons.add(model.x[0,177] == 0) gives new solution
        # model.cons.add(model.x[2,188] == 0) adding makes infeasible

        # model.cons.add(model.x[0,194] == 0) gives new solution
        # model.cons.add(model.x[1,239] == 0) adding makes infeasible

        # for checking prob-2-300|300-mm-2_2.json
        # model.cons.add(model.x[0,288] == 0) gives new solution
        # model.cons.add(model.x[1,257] == 0) adding makes infeasible

        # for checking prob-2-300|300|300|300|300-lllll-5_2.json
        # model.cons.add(model.x[1,262] == 0) #gives new solution but with objective value 1423 instead of 1422
        pass

    # Define objective
    def objective_rule(model):
        return sum(model.x[s, i] for s, i in model.SI)

    model.obj = pyomo.Objective(rule=objective_rule, sense=pyomo.minimize)

    return model


def solve_model(model: pyomo.ConcreteModel(), solver_str=DEFAULT_SOLVER, verbose=False):
    assert solver_str in ["cbc", "cplex_direct", "plpk", "glpk"]
    print(f"Solving model..  using {solver_str}")
    # Solve model
    solver = pyomo.SolverFactory(solver_str)
    solver.solve(model, tee=verbose)


def retrieve_solution(
    model: pyomo.ConcreteModel(), Y_list: list[PointList], verbose=False
) -> list[PointList]:
    # Retrieve solution
    xVal = []
    Y_generator_list = []
    for s in model.S:
        Y_generator = []
        if verbose:
            print(f"{s=}")
        for i in model.I_s[s]:
            if verbose:
                print("    x[{}]={}".format(i, pyomo.value(model.x[s, i])))
            if pyomo.value(model.x[s, i]) == 1:
                Y_generator.append(Y_list[s][i])
            elif math.isclose(pyomo.value(model.x[s, i]), 1):
                if verbose:
                    print(f"  math.isclose() rounding used..")
                Y_generator.append(Y_list[s][i])
        Y_generator_list.append(PointList(Y_generator))

    return Y_generator_list


def retrieve_solution_covering(
    model: pyomo.ConcreteModel(), Y_list: list[PointList], verbose=False
) -> list[PointList]:
    # Retrieve solution

    Y_generator_list = []

    Y_chosen_dict = {
        s: {i for i in model.I_s[s] if math.isclose(pyomo.value(model.x[s, i]), 1)}
        for s in model.S
    }

    # for (j,c) in model.JC:
    # incident_s_ys = [(s,ys) for (s,ys) in enumerate(Yn_comb_reduced_dict[j][c]) if ys is not None]
    # print('    z[{}]={}'.format((j,c), pyomo.value(model.z[j, c])))

    return Y_chosen_dict


def display_solution(
    Y_list: list[PointList], Y_generator_list: list[PointList], verbose=True, plot=True
):
    # Display solution
    if verbose:
        total_standard, total_generator = 0, 0
        if verbose == "all":
            print("_" * 55)
        for s, _ in enumerate(Y_list):
            total_standard += len(Y_list[s])
            if verbose == "all":
                print(f"|Y{s}| = {len(Y_list[s])}")
                print(f"|G{s}| = {len(Y_generator_list[s])}")
            total_generator += len(Y_generator_list[s])
        print("_" * 55)
        print(f"SUM|Y| = {total_standard}")
        print(f"SUM|G| = {total_generator}")
        print("_" * 55)

    if plot:
        plt.style.use("ggplot")
        Yn_all = methods.MS_sum(Y_list)
        Yn_all.plot("Y1n + Y2n +...", color="gray")

        for s, Y in enumerate(Y_list):
            Y.plot(l=f"Y{s}", marker="o")
        for s, Y_generator in enumerate(Y_generator_list):
            Y_generator.plot(l=f"G{s}", marker=f"{s+1}")

        Yn = methods.MS_sequential_filter(Y_list)
        Yn.plot(l="Yn")

        Yn_hat = methods.MS_sum(Y_generator_list)
        # assert N(Yn_hat.removed_duplicates()) == N(Yn.removed_duplicates())
        Yn_hat.plot(l="G Yn", marker="3")

        plt.show()


def solve_MGS_instance(Y_list: list[PointList], verbose="all", plot=False):
    model = build_model(Y_list)
    solve_model(model)
    Y_MIN_LIST = retrieve_solution(model, Y_list)
    if verbose or plot:
        display_solution(Y_list, Y_MIN_LIST, verbose=verbose, plot=plot)

    return Y_MIN_LIST


"""
SimpleFilter(Y_N^1, ..., Y_N^m)
   Input: Y^1, ..., Y^m
   Output: Y = {(y_1, I_1), ..., (y_|Y|, I_|Y|)} where 
    y_j = jth ND point and I_j = {(i_1, ..., i_m), ...} where 
    a vector (i_1, ..., i_m) equal the index from each 
    subproblem such that the MS gives y_j
      
   for (k = 1 to l) do I_k = {(k)} 
   Y = {(y^1_1, I_1), ..., (y^1_|Y^1|, I_|Y^1|)}
   for (s = 1 to m-1) do
      Y := Filter(Y, Y^(s+1))  // MS of Y, Y_N^(s+1) and add index
   next
   return (Y, I)
end

Filter(Y, Z)
   Q = {}   // contains pairs (y_i, I_i)
   for k = 1 to |Y|
      for t = 1 to |Z|
         q = y_k + z_t    
         Q := addPoint(q, Q, I_k, t) // add and update non dom set + add index (ties are updated too)
      next
   next
   return Q
end
"""


def SimpleFilterSub(Y, Ys):
    Y_ms = []

    Yn_points = methods.ND_pointsSum2_wrapper(Y, Ys).points
    # Build KDTree

    tree = KDTree(np.array([y.val for y in Yn_points]))
    # Define a tolerance for "very close"
    tolerance = 1e-3

    for k, y in enumerate(Y):
        for t, ys in enumerate(Ys):
            y_new = y + ys
            y_new.i = y.i + ys.i
            if tree.query_ball_point(y_new.val, tolerance):
                Y_ms.append(y_new)
            # if y_new in Yn_points:
            # Y_ms.append(y_new)

    # Yn_points = set(methods.N(PointList(Y_ms)).points)

    # Y_ms_N = PointList([y for y in Y_ms if y in Yn_points])

    Y_ms_N = PointList(Y_ms)
    Y_ms_N.points = Y_ms
    assert len(Y_ms_N) != 0
    return Y_ms_N


def SimpleFilter(Y_list):
    """
    input: list of PointList
    output: nondominated points of Minkowski sum of sets Y_list
    """

    # add index to each point
    for Ys in Y_list:
        for i, y in enumerate(Ys):
            y.i = [i]

    for Ys in Y_list:
        for y in Ys:
            if not hasattr(y, "i"):
                raise ValueError(f"Point {y} in {Ys} does not have index 'i'.")
    # Y_dict = {y:i for i,y in enumerate(Y_list[0]}
    Yn = Y_list[0]

    for s in range(1, len(Y_list)):
        print(f"{len(Yn),len(Y_list[s]),s=}")
        Yn = SimpleFilterSub(Yn, Y_list[s])

    # def get_dict():
    if True:
        C_dict = {y: [] for y in Yn}
        for y in Yn:
            C_dict[y].append(tuple(y.i))
        # Yn_dict_old = {y:[tuple(yn.i) for yn in Yn if yn == y] for y in Yn}
        # DONE: speed up the above <15-07-24, yourname> #

        # return C_dict
    # C_dict = get_dict()

    return PointList(Yn), C_dict
