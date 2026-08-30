import numpy as np
import shapely
from src.asmo.classes.problem import CVRP, SolutionList
from src.asmo.classes.asmo import ASMO, Bound
from src.asmo.classes.plotter import Plotter
from src.asmo.classes.pointsets import Point, PointList
from src.asmo.utils.timing import timeit, time_object
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
check_dominated_A_by_B = timeit(check_dominated_A_by_B)


def setup_asmo_example(preloaded_solutions=True, SMALL_TEST=True):
    if False:
        p1 = CVRP("CVRP_neg_0_16_8_35_NEW.lp")
        p2 = CVRP("CVRP_neg_0_16_8_35_NEW.lp")
        p3 = CVRP("CVRP_neg_0_16_8_35_NEW.lp")
        p4 = CVRP("CVRP_uni_0_16_8_35_NEW.lp")
        p5 = CVRP("CVRP_pos_0_16_8_35_NEW.lp")
        p4 = CVRP("newest_lp_file.lp")
        p2 = CVRP("newest_lp_file.lp")
        p3 = CVRP("newest_lp_file.lp")
        p5 = CVRP("newest_lp_file.lp")

    if False:
        p2 = CVRP("E-n22-k4.lp")
        p3 = CVRP("E-n22-k4.lp")
        p4 = CVRP("E-n22-k4.lp")
        p4 = CVRP("E-n22-k4.lp")
        p5 = CVRP("E-n22-k4.lp")
        p6 = CVRP("E-n22-k4.lp")

    # CVRP test
    if False:
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

    # WARNING: THIS ONE WORKS
    if False:
        P = [
            CVRP("CVRP_n13_k6_c1_r120_center_easy.lp"),
            CVRP("CVRP_n14_k6_c1_r120_center_easy.lp"),
            CVRP("P-n19-k2_easy.lp"),
        ]
    if False:
        P = [
            CVRP("CVRP_n13_k6_c1_r120_center_easy.lp"),
            CVRP("CVRP_n14_k6_c1_r120_center_easy.lp"),
            CVRP("A-n33-k5_easy.lp"),
        ]
    if SMALL_TEST:
        P = [
            CVRP("CVRP_n13_k6_c1_r120_center_easy.lp"),
            CVRP("CVRP_n13_k6_c1_r120_center_easy.lp"),
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


ASMO = time_object(ASMO)
CVRP = time_object(CVRP)
Bound = time_object(Bound)
shapely = time_object(shapely)


def test_comp_study_single(SMALL_TEST, strategy_search_method):
    CVRP_example = True
    preload = False
    if CVRP_example:
        asmo = setup_asmo_example(preloaded_solutions=preload, SMALL_TEST=SMALL_TEST)

    for p in asmo.P:
        p.solutions_preloaded = preload

    asmo.statistics["filename"] = "test"

    if CVRP_example:  # CVRP example
        asmo.statistics["filename"] = "test_cvrp"
        comp_study_single(
            asmo=asmo,
            plot_results=True,
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


if __name__ == "__main__":

    test_comp_study_single(SMALL_TEST=True, strategy_search_method="phase1")
    test_comp_study_single(SMALL_TEST=True, strategy_search_method="bbm")
