import numpy as np
import time
import copy
import shapely
from src.asmo.classes.problem import CVRP, SolutionList, AsmoPoint
from src.asmo.classes.asmo import ASMO, Bound
from src.asmo.classes.geom import Polygon, SearchArea
from src.asmo.classes.plotter import Plotter
from src.asmo.classes.pointsets import Point, PointList, MinkowskiSumProblem
from src.asmo.utils.upper_envelope import upper_envelope
from src.asmo.utils.mspMethods import induced_UB
from functools import reduce
from src.asmo.utils.timing import timeit, time_object, print_timeit
from src.asmo.utils.mspMethods import N
import src.asmo.classes.problem as problemModule
import src.asmo.classes.asmo as asmoModule
import src.asmo.classes.geom as geomModule
import src.asmo.classes.plotter as plotterModule
from comp_study import comp_study_single
from src.asmo.utils.fast_A_dominated_by_B import check_dominated_A_by_B
from src.asmo.utils import mspMethods

problemModule = timeit(problemModule)
mspMethods = timeit(mspMethods)
asmoModule = timeit(asmoModule)
geomModule = timeit(geomModule)
plotterModule = timeit(plotterModule)
upper_envelope = timeit(upper_envelope)
check_dominated_A_by_B = timeit(check_dominated_A_by_B)


def setup_asmo_example(preloaded_solutions=True):
    p1 = CVRP("CVRP_neg_0_16_8_35_NEW.lp")
    p2 = CVRP("CVRP_neg_0_16_8_35_NEW.lp")
    p3 = CVRP("CVRP_neg_0_16_8_35_NEW.lp")
    p4 = CVRP("CVRP_uni_0_16_8_35_NEW.lp")
    p5 = CVRP("CVRP_pos_0_16_8_35_NEW.lp")
    # p1 = CVRP("E-n22-k4.lp")
    # p2 = CVRP("E-n22-k4.lp")
    p4 = CVRP("newest_lp_file.lp")
    p2 = CVRP("newest_lp_file.lp")
    p3 = CVRP("newest_lp_file.lp")
    p5 = CVRP("newest_lp_file.lp")
    # p4 = CVRP("P-n16-k8.lp")
    # p1 = CVRP("E-n22-k4.lp")
    # p4 = CVRP("newest_lp_file.lp")

    if False:
        p2 = CVRP("E-n22-k4.lp")
        p3 = CVRP("E-n22-k4.lp")
        p4 = CVRP("E-n22-k4.lp")
        p4 = CVRP("E-n22-k4.lp")
        p5 = CVRP("E-n22-k4.lp")
        p6 = CVRP("E-n22-k4.lp")

    # CVRP test
    if True:
        # set CVRP.instance_dir = "./instances/cvrp/IP/"
        CVRP.instance_dir = "./instances/cvrp/max_min/"
        p1 = CVRP("CVRP_n22_k3_generated_easy.lp")
        p2 = CVRP("CVRP_n22_k4_c5_r120generated_easy.lp")
        p1 = CVRP("P-n19-k2_easy.lp")
        p3 = CVRP("P-n19-k2_easy.lp")
        p4 = CVRP("P-n19-k2_easy.lp")
        p1 = CVRP("CVRP_n14_k6_c1_r120_center_easy.lp")
        p11 = CVRP("CVRP_n14_k6_c1_r120_center_easy.lp")
        p2 = CVRP("CVRP_n14_k6_c1_r120_center_easy.lp")
        p4 = CVRP("CVRP_n14_k6_c1_r120_center_easy.lp")
        p4 = CVRP("CVRP_n14_k6_c1_r120_center_easy.lp")
        p5 = CVRP("CVRP_n14_k6_c1_r120_center_easy.lp")
        p6 = CVRP("CVRP_n14_k6_c1_r120_center_easy.lp")
        p7 = CVRP("CVRP_n14_k6_c1_r120_center_easy.lp")
        # p2 = CVRP("CVRP_n22_k3_generated_easy.lp")
        # p3 = CVRP("CVRP_n20_k2_c5_r60generated_easy.lp")
        # p3 = CVRP("CVRP_n22_k3_generated_easy.lp")
        # p4 = CVRP("CVRP_n22_k3_generated_easy.lp")
        p3 = CVRP("CVRP_n20_k2_c5_r60generated_easy.lp")
        p33 = CVRP("CVRP_n20_k2_c5_r60generated_easy.lp")

        # CVRP_n14_k6_c1_r120_center_easy.lp has many un-convex regions
        p2 = CVRP("CVRP_n14_k6_c1_r120_center_easy.lp")
        # CVRP_n14_k6_c5_r120_center_easy.l
        p2 = CVRP("CVRP_n14_k6_c5_r120_center_easy.lp")
        # CVRP_n14_k4_c1_r60_center_easy.lp
        # p3 = CVRP("CVRP_n14_k4_c1_r60_center_easy.lp")
        # cvrp_n13_k6_c8_r60_center_easy.lp
        # CVRP_n13_k6_c1_r120_center_easy.lp
        # convex: CVRP_n14_k6_c1_r120_center_easy.lp.
        p1 = CVRP("CVRP_n14_k6_c1_r120_center_easy.lp")
        peasy = CVRP("CVRP_n11_k2_c3_r120generated_easy.lp")
        peasy1 = CVRP("CVRP_n11_k2_c3_r120generated_easy.lp")

    P = [p1, p2, p3]
    P = [p1, p2, p4]
    P = [p1, p2, p3, p4]
    # working example 2 points skipped
    P = [
        CVRP("CVRP_n14_k6_c1_r120_center_easy.lp"),
        CVRP("CVRP_n14_k6_c5_r120_center_easy.lp"),
        CVRP("CVRP_n20_k2_c5_r60generated_easy.lp"),
        CVRP("CVRP_n14_k6_c1_r120_center_easy.lp"),
    ]

    # WARNING: THIS ONE WORKS
    if True:
        P = [
            CVRP("CVRP_n13_k6_c1_r120_center_easy.lp"),
            # CVRP("CVRP_n14_k6_c1_r120_center_easy.lp"),
            # CVRP("CVRP_n14_k6_c1_r120_center_easy.lp"),
            CVRP("CVRP_n14_k6_c1_r120_center_easy.lp"),
            CVRP("P-n19-k2_easy.lp"),
        ]
    if True:
        P = [
            CVRP("CVRP_n13_k6_c1_r120_center_easy.lp"),
            # CVRP("CVRP_n14_k6_c1_r120_center_easy.lp"),
            CVRP("CVRP_n14_k6_c1_r120_center_easy.lp"),
            CVRP("A-n33-k5_easy.lp"),
            # CVRP("P-n19-k2_easy.lp"),
        ]

    # P = [p3, p6, p7]
    # P = [p1, p2, p3, p4, p5, p6]
    # P = [p1, p2, p3, p4, p5]
    # P = [p3,p4]
    asmo = ASMO()

    for i, p in enumerate(P):

        if preloaded_solutions:
            p.preload_solutions()
        else:
            p_copy = CVRP(p.name)
            p_copy.epsilon_constraint()
            print(
                f"preloaded solutions for {p.name} saved to {CVRP.Yn_solution_dir + p.name + '_preloaded.json'}"
            )
            p_copy.hY.save_json(CVRP.Yn_solution_dir + p.name + "_preloaded.json")
            p.Yn = p_copy.hY
        p.name = p.name + "_" + str(i)
        # p.Yn = SolutionList.from_json("")
        # p.Yn = SolutionList.from_json("../instances/cvrp/Yn/CVRP_neg_0_16_8_35_NEW.lp_test.json")
        asmo.load_subproblem(p)

    return asmo


def setup_article_example():
    return


def setup_asmo_example_redundant():
    p1 = CVRP("CVRP_neg_0_16_8_35_NEW.lp")
    p2 = CVRP("CVRP_neg_0_16_8_35_NEW.lp")
    p3 = CVRP("CVRP_neg_0_16_8_35_NEW.lp")
    p4 = CVRP("CVRP_neg_0_16_8_35_NEW.lp")
    # P = [p1, p2, p3]
    P = [p1, p2]
    # P = [p3,p4]

    asmo = ASMO()
    for i, p in enumerate(P):
        p.preload_solutions()
        # p.solutions_preloaded = True # set flag
        p.name = p.name + "_" + str(i)

    Y1_coords = np.array([(10, 100), (80, 95), (94, 94), (95, 80), (100, 10)]) * 10
    Y1_se_coords = np.array([(10, 100), (100, 10)]) * 10

    # Y1 = PointList([Point((x,y)) for x,y in Y1_coords])
    # X1 = SolutionList([Solution(y.val, 'p1', 'None') for y in Y1])

    Y1 = PointList.from_json("./instances/subproblems_local_sets/sp-2-100-m_3.json")
    Y1 = PointList.from_json("./instances/subproblems_local_sets/sp-2-10-m_3.json")
    Y1 = PointList.from_json(
        "./instances/subproblems_local_sets/sp-2-50-l_3.json"
    )  # * 2
    Y1 = PointList([y * Point((8, 1)) for y in Y1])
    Y2 = PointList.from_json("./instances/subproblems_local_sets/sp-2-100-u_2.json")

    Y2 = PointList.from_raw("../bi_objective_p_median/points.raw")
    Y2 = N(Y2)

    # Y1 = Y2
    Y1 = PointList.from_json("./instances/subproblems_local_sets/sp-2-10-m_3.json")
    Y2 = PointList.from_json("./instances/subproblems_local_sets/sp-2-10-u_3.json")

    if False:  # example from MSPGeerators
        # Y2 <- matrix(  # n x p matrix
        #    c(0, 7,
        #      3, 6,
        #      4, 5,
        #      6, 4,
        #      9, 3,
        #      10, 2,
        #      17, 1,
        #      20, 0
        #    ), ncol = p, byrow = T)
        Y2 = PointList(
            (
                Point(y)
                for y in (
                    (0, 7),
                    (3, 6),
                    (4, 5),
                    (6, 4),
                    (9, 3),
                    (10, 2),
                    (17, 1),
                    (20, 0),
                )
            )
        )
        scaling = 1
        Y2 = N(Y2) * scaling
        # Y1 <- matrix(  # n x p matrix
        #    c(12-12, 4,
        #      14-12, 3,
        #      16-12, 2
        Y1 = PointList(
            (
                Point(y)
                for y in (
                    (0, 4),
                    (2, 3),
                    (4, 2),
                )
            )
        )
        Y1 = N(Y1) * scaling

    if True:

        Y1 = SolutionList.from_json(
            # f"../instances/cvrp/eksempel_bi_cvrp16.json"
            # f"../instances/cvrp/max_min/Yn/p-n22-k8.json"
            "./instances/cvrp/max_min/Yn/newest_lp_file.json"
        ).as_pointlist()
        Y2 = SolutionList.from_json(
            "./instances/cvrp/max_min/Yn/newest_lp_file_fjernes.json"
            # f"../instances/cvrp/eksempel_bi_cvrp.json",
            # f"../instances/cvrp/newest_lp_file.json",
            # "../instances/cvrp/max_min/Yn/weierstrass_rastrigin_pareto_correct.json"
            # "../instances/cvrp/max_min/Yn/weierstrass_rastrigin_pareto_70_interpolated.json"
        ).as_pointlist()
        # Y2 = PointList.from_json("../instances/subproblems_local_sets/sp-2-50-u_8.json")
        Y1 = N(Y1)  # * 2
        Y2 = N(Y2)  # * 13  # 13

        # Y1 = copy.deepcopy(Y2)
        # Round all Y2 to integers
        # Y2 = PointList([Point((round(y.val[0]), round(y.val[1]))) for y in Y2])

    if False:  # Sunes eksempel

        Y2 = PointList.from_raw("./instances/cvrp/sune_eksempel2.raw") * Point(
            (4, -1)
        ) + PointList((Point((0, 40000)),))
        Y2 = N(Y2)

        if True:

            Y1 = PointList.from_raw("./instances/cvrp/sune_eksempel2.raw") * Point(
                (3, -1)
            ) + PointList((Point((0, 41000)),))
            Y1 = N(Y1)

    def read_max_and_normalise(filename: str):
        Y = PointList.from_raw(filename)
        Y = Y * (-1)
        y_I = Y.get_ideal()
        Y = Y - PointList(
            (y_I),
        )
        return N(Y)

    if False:  # fixed/variable cost problems
        # Y3 = PointList.from_raw('../../bs-ip-code/tests/instances/modc_Y_save_test_0.raw')
        # y_I = Y3.get_nadir()
        # Y3 = Y3 *(-1)  + PointList((y_I), )
        # Y3 = Y3 + PointList((Point((1000,1000)), ))
        # Y3 = Y3*2
        #
        # Y4 = PointList.from_raw('../../bs-ip-code/tests/instances/modc_Y_save_test_1.raw')
        # y_I = Y3.get_nadir()
        # Y4 = Y4 * (-1)  + PointList((y_I), )
        # Y4 = Y4 + PointList((Point((1000,1000)), ))
        # Y4 = Y4 *2

        # Y1 = N(Y3)
        # Y2 = N(Y4)
        Y1 = read_max_and_normalise(
            "../../bs-ip-code/tests/instances/modc_Y_save_test_0.raw"
        )
        Y2 = read_max_and_normalise(
            "../../bs-ip-code/tests/instances/modc_Y_save_test_1.raw"
        )
        Y2 = read_max_and_normalise(
            "../../bs-ip-code/tests/instances/modc_Y_save_test_blocks_1.raw"
        )
        Y2 = read_max_and_normalise(
            "../../bs-ip-code/tests/instances/modc_Y_save_test_blocks_0.raw"
        )
        Y1 = read_max_and_normalise(
            "../../bs-ip-code/tests/instances/modc_Y_save_test_blocks.raw"
        )
        Y2 = read_max_and_normalise(
            "../../bs-ip-code/tests/instances/modc_Y_save_test_blocks.raw"
        )

    Plot = Plotter()
    Y1.plot(Plot.axs[0], color="blue")
    Y2.plot(Plot.axs[0], color="red")
    Plot.save("tests/test_asmo_setup_redundant.pdf")
    #
    # Y2 = lex_sort(Y2)
    # Y2 = PointList(sorted(Y2.points))*Point((1,0.9)) #WHY WAS THIS HERE... :(
    Y2 = PointList(sorted(Y2.points))

    X1 = SolutionList.from_pointlist_dummy(Y1, p1.name)
    X2 = SolutionList.from_pointlist_dummy(Y2, p2.name)

    X1 = SolutionList(sorted(X1, key=lambda x: x.val[0]))
    X2 = SolutionList(sorted(X2, key=lambda x: x.val[0]))
    # X1 = SolutionList.from_coords_dummy(Y1_coords, p1.name)

    p1.Yn = X1
    p1.Yse = X1.get_supported()
    p2.Yn = X2
    p2.Yse = X2.get_supported()
    p3.Yn = X2
    p3.Yse = X2.get_supported()
    # p2.Yse = SolutionList(sorted(X2, key = lambda x: x.val[0]))
    # p1.Yse = SolutionList.from_coords_dummy(Y1_se_coords, p1.name)

    for i, p in enumerate(P):
        asmo.load_subproblem(p)

    # load a third problem
    from copy import deepcopy

    if True:  # load p3
        p3.preload_solutions()
        p3.Yn = deepcopy(p2.Yn)
        p3.name = p3.name + "_copy"
        p3.Yse = deepcopy(p2.Yse)
        asmo.load_subproblem(p3)

        if False:
            p4.preload_solutions()
            p4.Yn = deepcopy(p2.Yn)
            p4.name = p3.name + "_copy2"
            p4.Yse = deepcopy(p2.Yse)
            asmo.load_subproblem(p4)
    if __debug__:  # check that all points in Yn are integer vectors
        for p in asmo.P:
            for y in p.Yn:
                for yi in y.val:
                    if yi % 1 != 0:
                        raise ValueError(f"Point {y} in Yn is not an integer vector")

    return asmo


def test_asmo_setup():

    # from src.utils.minimumGenerator import solve_MGS_instance
    from src.utils.fastMinimumGenerator import algorithm2

    # asmo = setup_asmo_example()
    # for p in asmo.P:
    # print(len(p.Yn))

    asmo = setup_asmo_example_redundant()

    Y_list = [p.Yn.as_pointlist() for p in asmo.P]
    # add index to each point

    msp = MinkowskiSumProblem(Y_list)
    msp.filename = "test_redundent"
    # = algorithm2(msp)

    # get reduced
    asmo.get_minimum_generator()

    mgs, Yn = algorithm2(msp)
    # print(mgs_statistics)
    # print(mgs_statistics)

    for Yns, Ygs in zip(msp.Y_list, mgs.Y_list):
        print(f"Yns: {len(Yns)}, Ygs: {len(Ygs)}")
    # solve_MGS_instance(Y_list,verbose='all')
    for p in asmo.P:
        print(f"Subproblem {p.name},  {len(p.Y_mgs)}:")

    plotter = Plotter()
    for s, p in enumerate(asmo.P):
        p.initialize_subproblem()
        plotter.plot(p.Yn, name=f"Yn{s+1}")
        plotter.plot(p.Yse, name=f"Yn{s+1}", marker="x")
        plotter.plot(p.Y_mgs, name=f"Yg{s+1}", marker="x", s=2)
        plotter.plot(p.L, name=f"L{s+1}")
        plotter.plot(p.U, name=f"U{s+1}")
        # for n in p.T.T.queue:
        # plotter.plot(n.R, name=f"tY{s+1}", marker='o')

    plotter.add_legend()
    plotter.save("tests/test_asmo_setup.pdf")


def test_matrix_plot():
    # import pointlist and test the ND filters from the msp module
    # from src.classes.pointclasses import PointList
    # from src.classes.pointclass import PointList, Point
    from src.asmo.utils.mspMethods import N
    from src.asmo.utils.matrixPlots import matrix_plot

    # import src.utils.minimumGenerator as mgs

    # Create a PointList object with some test data

    # Creat a pointlist containing 100 2d points in the range 0-100 which are negatively corrolated
    np.random.seed(0)
    points = np.random.rand(100, 2) * 100

    Y1 = PointList([Point((p[0], p[1])) for p in points]) * 2
    points = np.random.rand(100, 2) * 100
    Y2 = PointList([Point((p[0], p[1])) for p in points]) * 2

    Y1 = N(Y1)
    Y2 = N(Y2)

    Y = Y1 + Y2

    # Yn = N(Y)

    matrix_plot(
        Y1,
        Y2,
        "../results/plots/tests/matrix_plot.pdf",
        point_labels=False,
        matrix_only=False,
        plot_mapping=True,
        figsize=(7, 3),
    )


def test_asmo_matrix_plot():
    from src.asmo.utils.matrixPlots import matrix_plot

    asmo = setup_asmo_example_redundant()
    # asmo = setup_asmo_example()
    asmo.P = asmo.P[:2]
    assert len(asmo.P) == 2, "not implemented for 2 < S"

    Y1 = PointList([Point(y.val) for y in asmo.P[0].Yn])
    Y2 = PointList([Point(y.val) for y in asmo.P[1].Yn])

    Y1 = Y1[::8]
    Y2 = Y2[::8]
    print(len(Y1), len(Y2))

    matrix_plot(
        Y1,
        Y2,
        # "../results/plots/tests/matrix_plot_redundent.pdf",
        "matrix_plot_redundent.pdf",
        point_labels=True,
        matrix_only=False,
        plot_mapping=True,
        figsize=(7, 3),
    )


def test_asmo_solver_initialization():
    asmo = setup_asmo_example()

    # initiate each subproblem
    for p in asmo.P:
        p.initialize_subproblem()
        p.partition_search_area(p.Yse)


def test_asmo_solver_refine_upper_bound():
    # single refine call
    asmo = setup_asmo_example()

    # initiate each subproblem
    for p in asmo.P:
        p.initialize_subproblem()
        p.partition_search_area(p.Yse)

    asmo._set_global_upper_bound()

    p = asmo.P[1]
    L_hat = asmo._get_L_hat(p)
    # plot upper bound before and after
    plotter = Plotter()
    asmo._set_incumbent()  # sets asmo.hY
    # get local nadir point of incumbent
    local_nadir_points = induced_UB(asmo.hY.as_pointlist())
    if False:
        plotter.plot(p.U, color="red", name="U")
        plotter.plot(p.Yn, color="gray", name="Yn", s=0.1)
        plotter.plot(p.Yse, color="black", name="Yse", s=0.1)
        plotter.plot(asmo.U, color="orange", name="U", linewidth=0.3)
        plotter.plot(asmo.hY, color="green", name="incumbent", marker="x")
        local_nadir_points.plot(ax=plotter.axs[0], color="red", l="N", marker="x")
    U_minus_L_hat = [
        ((-L_hat) + AsmoPoint(u.val)).extend_upper(10000)
        for u in local_nadir_points.points
    ][::1]

    fast_Ugs = Bound(upper_envelope([L.geom for L in U_minus_L_hat]))
    fast_Ugs = L_hat
    print(fast_Ugs)
    for u_minus_l_hat in U_minus_L_hat:
        plotter.plot(u_minus_l_hat, name="U - L_hat", linewidth=0.2)
    plotter.plot(
        fast_Ugs,
        # p.U.merge_upper_bounds(fast_Ugs),
        color="blue",
        name="U - L_hat (fast)",
        linewidth=0.1,
    )
    plotter.plot(
        L_hat,
        name=r"$\hat{\mathcal{L}}$",
        color="blue",
        linewidth=0.2,
    )

    # refine single upper bound
    # asmo._reduce_upper_bound(p=p)
    # update all
    asmo._reduce_upper_bounds()

    if False:
        plotter.plot(p._Ugs, color="orange", name="L", linewidth=0.2)
        plotter.plot(p.U, color="red", name="L", linewidth=0.2)
    # plotter.plot(res, color="green", name="L", linewidth=0.5)

    plotter.add_legend()
    all_geoms = shapely.union_all([p.Yse.get_geoms(), asmo.hY.get_geoms()])
    # plotter.zoom(geomobject=all_geoms, buffer=30, buffer_relative=True)
    plotter.zoom(geomobject=fast_Ugs.geom, buffer=100, buffer_relative=True)
    plotter.save("tests/test_asmo_solver_refine_upper_bound.pdf")


def test_asmo_solver_refine_upper_bound_compare():
    # single refine call
    asmo = setup_asmo_example()

    # initiate each subproblem
    for p in asmo.P:
        p.initialize_subproblem()
        p.partition_search_area(p.Yse)

    asmo._set_global_upper_bound()

    p = asmo.P[0]

    # plot upper bound before and after
    plotter = Plotter()
    plotter.plot(p.U, color="red", name="U")
    plotter.plot(
        p.Yn,
        color="gray",
        name="Yn",
    )
    plotter.plot(p.Yse, color="black", name="Yse")

    plotter.plot(asmo.U, color="orange", name="U", linewidth=0.3)

    U_before = copy.deepcopy(p.U)
    # refine single upper bound
    time_slow = time.time()
    asmo._reduce_upper_bound(p=p, version="slow")
    plotter.plot(p.U, color="orange", name="L", linewidth=0.5)
    time_slow = time.time() - time_slow

    p.U = U_before  # reset upper bound
    time_fast = time.time()
    asmo._reduce_upper_bound(p=p, version="fast")
    time_fast = time.time() - time_fast
    plotter.plot(p.U, color="green", name="L", linewidth=0.3)

    print(f"{time_slow=}")
    print(f"{time_fast=}")
    # update all
    # asmo._reduce_upper_bounds()

    plotter.add_legend()
    plotter.save("tests/test_asmo_solver_refine_upper_bound.pdf")


def test_asmo_object():

    asmo = setup_asmo_example()
    # asmo = setup_asmo_example_redundant()

    # initial partition
    for p in asmo.P:
        p.initialize_subproblem()
        p.hY = p.Yn
        p.partition_search_area(p.Yse)

    def solve_p_n_times(p: "Problem", n=10, search_method="bbm"):
        """Solve the subproblem p n times"""
        p.search_method = search_method
        for _ in range(n):
            n = p.T.getNode()
            if n is None:
                break
            p.search_node(n)
            p._update_bounds_from_node(n)
            for new_node in n.T:
                p.T.addNode(new_node)

    solve_p_n_times(asmo.P[0], n=4, search_method="bbm")
    solve_p_n_times(asmo.P[0], n=4, search_method="phase1")
    solve_p_n_times(asmo.P[1], n=4, search_method="bbm")
    solve_p_n_times(asmo.P[1], n=4, search_method="phase1")

    # test set bounds
    asmo._set_incumbent()
    asmo._set_global_lower_bound()
    asmo._set_global_upper_bound()

    if False:  # add ROI
        # add a region of interest from asmo.hY
        W = (asmo.hY.get_nadir_point()[1] + asmo.hY.get_ideal_point()[1]) / 2
        ROI = asmo.hY.region_of_interest(W, gamma=0.0)
        asmo.ROIr = ROI
    else:
        ROI = None

    ax_mosaic = [
        ["main"] * len(asmo.P),
        ["main"] * len(asmo.P),
        [f"1_{p.name}" for p in asmo.P],
        # [f"2_{p.name}" for p in asmo.P],
        # [f"3_{p.name}" for p in asmo.P],
    ]
    plotter = Plotter(ax_mosaic=ax_mosaic, figsize=(20, 20))

    plotter.axs["main"].set_title("Main")
    plotter.plot(asmo.L, ax="main", linestyle="dashed")
    plotter.plot(asmo.U, ax="main", linestyle="dashed")
    plotter.plot(asmo.hY, ax="main", color="gray")
    plotter.zoom(ax="main", geomobject=asmo.hY.get_geoms(), buffer=500)

    if ROI is not None:
        plotter.plot(ROI, ax="main", color="orange", alpha=0.5, name="ROI")
        plotter.axs["main"].set_title("Main with ROI")

    for s, p in enumerate(asmo.P):
        p.color = s + 2
        plotter.zoom(geomobject=p.Yse.get_geoms(), ax=f"1_{p.name}", buffer=500)
        plotter.axs[f"1_{p.name}"].set_title(f"Subproblem {s+1}")
        plotter.plot(p.L, color="blue", ax=f"1_{p.name}", alpha=0.5)
        plotter.plot(p.U, color="red", ax=f"1_{p.name}", alpha=0.5)
        plotter.plot(p.Yse, color="green", ax=f"1_{p.name}", alpha=0.5)
        plotter.plot(p.Yn, color="black", ax=f"1_{p.name}", alpha=0.5)

    # reduce bounds
    asmo._set_incumbent()
    # asmo._set_global_lower_bound()
    # asmo._set_global_upper_bound()
    for p in asmo.P:
        asmo._reduce_upper_bound(p=p)  # reduce upper bound for each subproblem
        asmo._reduce_lower_bound(p=p)  # reduce upper bound for each subproblem
    # asmo._reduce_upper_bounds()

    for s, p in enumerate(asmo.P):
        plotter.plot(p.L, color="blue", ax=f"1_{p.name}", alpha=0.5, linestyle="dashed")
        plotter.plot(p.U, color="red", ax=f"1_{p.name}", alpha=0.5, linestyle="dashed")

    if ROI:
        plotter.save("tests/test_asmo_object_roi.pdf")
    else:
        plotter.save("tests/test_asmo_object.pdf")


def test_asmo_solver_parts():

    asmo = setup_asmo_example()

    # initial generator bound update

    # initial partition
    for p in asmo.P:
        p.initialize_subproblem()
        p.partition_search_area(p.Yse)

    S = tuple(range(len(asmo.P)))
    plotter = Plotter(nrows=2, ncols=3)
    for s, p in enumerate(asmo.P):
        plotter.plot(p.L, color="blue", ax=(0, s))
        plotter.plot(p.U, color="red", ax=(0, s))
        plotter.plot(p.Yse, color="green", ax=(0, s))

    # main loop
    i = 0
    plotter.save(f"tests/test_asmo_solver_parts/{i}.pdf")

    while True:
        i += 1
        # select subproblem
        p = asmo.select_subproblem(i)
        if p is None:
            break
        # update node priority
        # asmo._update_node_priority(i)
        # search phase one
        n = p.T.getNode()

        if n is None:
            continue

        p._search_phase_one(n)

        p._update_bounds_from_node(n)

        p.partition_search_area(n.tY)
        print(i)


ASMO = time_object(ASMO)
CVRP = time_object(CVRP)
Bound = time_object(Bound)
shapely = time_object(shapely)


def plot_asmo(asmo, plotter):
    for s, p in enumerate(asmo.P):
        plotter.plot(p.L, color="blue", ax=(0, s), alpha=0.5)
        plotter.plot(p.U, color="red", ax=(0, s), alpha=0.5)
        plotter.plot(p.Yse, color="green", ax=(0, s))
        plotter.plot(p.Yn, color="black", ax=(0, s))
        # plotter.plot(p.T.T.queue[0].R, color="black", ax=(1,s), name=f"tY{s+1}")
        # plotter.plot(p.T.T.queue[0].R.tY, color="black", ax=(1,s), name=f"tY{s+1}")
    plotter.save(f"tests/asmo_plot.pdf")


def plot_node(n, plotter, ax):
    plotter.plot(n.R, color="black", ax=ax)
    plotter.plot(n.tY, color="green", ax=ax)
    plotter.plot(n.tU, color="red", ax=ax, linestyle="dashed")
    plotter.plot(n.tL, color="blue", ax=ax, linestyle="dashed")


def test_solve_development():

    np.random.seed(0)
    REDUCE_GENERATOR_BOUNDS = True

    asmo = setup_asmo_example()
    # asmo = setup_asmo_example_redundant()
    self = asmo

    plotter = Plotter(nrows=2, ncols=len(asmo.P))
    plotter.fig.set_size_inches(14 * 2, 12 * 2)
    # initialization step
    for p in self.P:
        p.initialize_subproblem()

    # set global upper bound
    self._set_global_upper_bound()  # U = ND_SUM(P[0].U + ... + P[S].U)

    plot_asmo(self, plotter)

    # reduce upper bounds
    if REDUCE_GENERATOR_BOUNDS:
        self._reduce_upper_bounds()  # updates p.U for each p in P

    for s, p in enumerate(self.P):
        plotter.plot(p.U, color="yellow", ax=(0, s))
        print(p.U.geom)
        p.plot_ax = s

    # plotter.save()

    # initate search tree for each subproblem
    for p in self.P:
        p.partition_search_area(p.hY)

    i = 0
    while p := self.select_subproblem(i):
        plotter = Plotter(nrows=2, ncols=len(asmo.P))
        plotter.fig.set_size_inches(14 * 2, 12 * 2)
        i += 1
        if p is None:
            break  # terminate

        # search brancing node and branch - updates n.p.hY and n.p.T
        n = p.T.getNode()

        assert n is not None, p.T.T.queue

        # check for pruning
        if n.check_if_empty(p.L, p.U) and False:
            print(f"SKIPPING empty node search area {n}:")
            i += 1
            continue

        p.search_method = "bbm"

        p.search_node(n)
        plot_node(n, plotter, ax=(0, p.plot_ax))
        plot_node(n, plotter, ax=(1, p.plot_ax))

        plotter.save(f"tests/test_solve_development_{i}.pdf")

        try:
            p._update_bounds_from_node(n)
        except:
            pass

        plotter.plot(p.L, color="blue", ax=(0, p.plot_ax), alpha=0.8)
        plotter.plot(p.U, color="red", ax=(0, p.plot_ax), alpha=0.8)

        plotter.save(f"tests/test_solve_development_{i}.pdf")

        for t in n.T:
            p.T.addNode(t)


def test_asmo_solver():

    np.random.seed(0)

    asmo = setup_asmo_example()
    from comp_study import file_to_asmo

    asmo = setup_asmo_example_redundant()
    asmo = file_to_asmo("./instances/msp/2obj/Lyngesen24-msp-2-50|50-ul-2_1.json")
    asmo = file_to_asmo("instances/msp/2obj/Lyngesen24-msp-2-50|50|50-ull-3_1.json")

    if __debug__:  # check that all points in Yn are integer vectors
        for p in asmo.P:
            for y in p.Yn:
                for yi in y.val:
                    if yi % 1 != 0:
                        raise ValueError(f"Point {y} in Yn is not an integer vector")
                    if yi < 0:
                        raise ValueError(
                            f"Point {y} in Yn is negative, should be positive"
                        )

    # set strategies
    asmo.get_minimum_generator()
    # asmo.strategy_subproblem_selection = "sequential"
    asmo.strategy_subproblem_selection = "alternating"
    asmo.strategy_reduction = "none"
    asmo.strategy_reduction = 1  # refine Ug^s every x iteration
    asmo.strategy_reduction = "first"
    asmo.strategy_refine_with_integer_gap = False  #

    if True:
        for p in asmo.P:
            p.search_method = "random"
            p.search_method = "bbm"
            p.search_method = "phase1"

    asmo.Yn = reduce(lambda hY1, hY2: (hY1 + hY2).N(), [p.Yn for p in asmo.P])
    ax_mosaic = [
        ["main"] * len(asmo.P),
        ["main"] * len(asmo.P),
        [f"1_{p.name}" for p in asmo.P],
        [f"2_{p.name}" for p in asmo.P],
        [f"3_{p.name}" for p in asmo.P],
    ]

    plotter = Plotter(ax_mosaic=ax_mosaic, figsize=(14 * 1, 12 * 1))

    # plotter.axs['main'].set_title("Main")

    # plotter padding between subplots
    plotter.fig.subplots_adjust(wspace=0.2, hspace=0.5)

    # set zoom level, a box containing all points
    all_sp_points = [y.geom for p in asmo.P for y in p.Yse]
    zoom = Polygon([y.geom for y in all_sp_points]).buffer(50)
    if False:
        plotter.plot(asmo.P[0].Yn, ax="main", color="gray", name="Yn", s=1)
        plotter.plot(asmo.P[1].Yn, ax="main", color="blue", name="Yn2", s=1)
        plotter.plot(asmo.P[0].Yse, ax="main", color="black", name="Yn", s=1)
        plotter.plot(asmo.P[1].Yse, ax="main", color="yellow", name="Yn2", s=1)
        plotter.save("/tests/test_asmo_solver/initial.pdf")
        # plotter.show()

    plotter.save("tests/test_asmo_solver/test_asmo_solver.pdf")
    asmo.print_strategy_config()

    if True:  # plot algorithm
        asmo.solve(plotter)
    else:
        asmo.solve(None)

    print_timeit()
    # hYn = ((asmo.P[0].hY + asmo.P[1].hY).N() + asmo.P[2].hY).N()
    # hYn = reduce(lambda x, y: (x + y).N(), [p.hY for p in asmo.P])
    hYn = asmo.hY
    print(len(hYn))
    Yn = reduce(lambda x, y: (x + y).N(), [p.Yn for p in asmo.P])
    print(len(Yn))

    plotter.plot(Yn, ax="main", color="red", name=f"|Yn|={len(Yn)}", s=6)
    plotter.plot(hYn, ax="main", color="blue", name=f"|hY|={len(hYn)}", s=3)

    subproblem_points = [len(p.Yn) for p in asmo.P]
    subproblem_points_generated = [len(p.hY) for p in asmo.P]
    mgs_points = [len(p.Y_mgs) for p in asmo.P]

    print(f"Subproblem points: {subproblem_points}, Total: {sum(subproblem_points)}")
    print(
        f"Subproblem points generated: {subproblem_points_generated}, Total: {sum(subproblem_points_generated)}"
    )
    print(f"MGS points: {mgs_points}, Total: {sum(mgs_points)}")

    # add three above plots to title
    plotter.axs["main"].set_title(
        f"Subproblem points: {subproblem_points}\n Generated: {subproblem_points_generated}\n MGS: {mgs_points}"
    )

    plotter.save("tests/test_asmo_solver/test_asmo_solver.pdf")

    print(len(Yn))
    print(len(hYn))
    assert len(hYn) == len(Yn)
    for y in Yn:
        assert AsmoPoint(y.val) in hYn.as_pointlist(), f"Point {y} in Yn not in hYn"



def test_asmo_fast_bound_ms():

    asmo = ASMO.from_json("./instances/msp/2obj/Lyngesen24-msp-2-50|50|50-ull-3_1.json")

    for p in asmo.P:
        p.initialize_subproblem()

    A = asmo.P[2].L
    B = asmo.P[1].L

    a_b_pairs = ASMO._ms_sum_upper_part_pairs(A, B, plot=True)

    a_b_pairs_all = [(a, b) for a in A.get_lines() for b in B.get_lines()]
    print(f"{len(a_b_pairs)=}, {len(a_b_pairs_all)=}")
    P = Plotter()

    for a, b in a_b_pairs[0:]:
        s = a + b
        P.plot(a + b)
        P.plot(SearchArea(s).get_upper_bound(), color="green")

    if True:
        for a, b in a_b_pairs:
            s = a + b
            P.plot(s, color="blue", lw=0.5)
            P.plot(
                SearchArea(s).get_upper_bound().extend_upper(100000),
                color="gray",
                lw=0.5,
            )

    if True:
        UE = upper_envelope(
            [
                SearchArea((a + b)).get_upper_bound().extend_upper(1000000)
                for a, b in a_b_pairs
            ],
            assume_full_coverage=False,
            x_min=0,
            x_max=20000,
        )
        P.plot(UE, color="red", name="UE", lw=0.3)

    # slow method
    if True:
        Upper_part = SearchArea(
            shapely.union_all([(a + b) for a, b in a_b_pairs])
        ).get_upper_bound()
        P.plot(Upper_part, color="orange", name="UE_slow", lw=0.2)
    # ]
    # )
    P.zoom(A.geom, buffer=6000)
    # UE = SearchArea(
    #     shapely.GeometryCollection([a + b for a, b in a_b_pairs])
    # ).get_upper_bound()
    # P.plot(UE, color="red", name="UE")
    P.save("tests/test_asmo_fast_bound_ms.pdf")


def test_comp_study_single():
    # asmo = setup_asmo_example_redundant()
    CVRP_example = False
    preload = True
    if CVRP_example:
        asmo = setup_asmo_example(preloaded_solutions=preload)

    else:
        # asmo = ASMO.from_json("instances/as-mo/Lyngesen24-msp-2-100|100|100-mmm-3_4")
        asmo = ASMO.from_json(
            "instances/msp/2obj/Lyngesen24-msp-2-100|100|100-mmm-3_4.json"
        )
        asmo = ASMO.from_json("instances/msp/2obj/Lyngesen24-msp-2-100|100-ul-2_1.json")
    for p in asmo.P:
        p.solutions_preloaded = preload

    asmo.statistics["filename"] = "test"

    # asmo = ASMO.from_json("./instances/msp/2obj/Lyngesen24-msp-2-100|100-ul-2_1.json")
    # asmo_1 = ASMO.from_json("./instances/msp/2obj/Lyngesen24-msp-2-100|100-mm-2_1.json")
    # asmo = ASMO.from_json("./instances/msp/2obj/Lyngesen24-msp-2-50|50|50-ull-3_1.json")

    # asmo.P[1] = asmo.preload_subproblem(1, asmo_1.P[1].name)
    # asmo.P[1].Yn = asmo_1.P[1].Yn

    # "Lyngesen24-msp-2-100|100-ul-2_1.json"

    if CVRP_example:  # CVRP example
        asmo.statistics["filename"] = "test_cvrp"
        comp_study_single(
            asmo=asmo,
            plot_results=False,
            csv_out="debug_run.csv",
            strategy_subproblem_selection="alternating",  # WARNING: This works ALTERNAING
            strategy_node_priority="largest",
            # strategy_reduction="all",
            # strategy_reduction=50,
            strategy_reduction=50,
            # strategy_reduction="none",
            strategy_update_node_priority="all",
            strategy_refine_method="slow",
            strategy_search_method="phase1",  # WARNING: works with phase1
            strategy_refine_with_integer_gap=False,  # WARNING: works with False
            preload_solutions=preload,
        )
    else:
        comp_study_single(
            asmo=asmo,
            plot_results=False,
            csv_out="debug_run.csv",
            # strategy_subproblem_selection="alternating",
            strategy_subproblem_selection="sequential",
            strategy_node_priority="largest",
            # strategy_reduction="all",
            # strategy_reduction=50,
            # strategy_reduction=50,
            strategy_reduction=100,
            strategy_update_node_priority="all",
            strategy_refine_method="smart",
            strategy_search_method="phase1",
            strategy_refine_with_integer_gap=False,
            preload_solutions=True,
        )

    if True:
        P = Plotter(len(asmo.P))
        for s, p in enumerate(asmo.P):
            ax = P.axs[s]
            Yn = p.Yn.as_pointlist()
            Yn.save_json(f"tests/test_comp_study_single_Yn_{s+1}_none.raw")
            hY = p.hY.as_pointlist()
            hY.save_json(f"tests/test_comp_study_single_hY_{s+1}_none.raw")
            Yn.plot(ax=ax, color="gray", l="Yn", s=1.3)
            hY.plot(ax=ax, color="blue", l="hY", s=0.1)
            print(f"Subproblem {s+1}: |Yn|={len(Yn)}, |hY|={len(hY)}")
            print(
                f"Subproblem {s+1} -dublicates: |Yn|={len(Yn.removed_duplicates())}, |hY|={len(hY.removed_duplicates())}"
            )
        P.save("tests/test_comp_study_single.pdf")

    print("test")


def test_comp_study_single_two():
    # Go throug and test if redundant solutions are found.
    # asmo = setup_asmo_example_redundant()
    asmo = setup_asmo_example(preloaded_solutions=True)
    for p in asmo.P:
        p.solutions_preloaded = True
    asmo.statistics["filename"] = "test"

    res1 = comp_study_single(
        asmo=asmo,
        plot_results=False,
        csv_out="debug_run.csv",
        strategy_subproblem_selection="sequential",
        strategy_node_priority="largest",
        # strategy_reduction="all",
        # strategy_reduction=50,
        strategy_reduction=300,
        strategy_update_node_priority="all",
        strategy_refine_method="fast",
        strategy_search_method="bbm",
        strategy_refine_with_integer_gap=False,
        preload_solutions=False,
    )
