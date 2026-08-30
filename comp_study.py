import logging
import os
import argparse
from time import perf_counter
from src.asmo.classes.asmo import ASMO
from src.asmo.classes.problem import CVRP, SolutionList
from src.asmo.classes.plotter import Plotter
from tqdm import tqdm
from src.asmo.classes.geom import Polygon
from src.asmo.classes.pointsets import PointList
import json, csv
from functools import reduce
from itertools import product
from src.asmo.utils.timing import (
    print_timeit,
    time_object,
    TIME_dict,
    COUNT_dict,
    reset_timeit,
)

logger = logging.getLogger("comp_study_single")
logging.basicConfig(filename="test_asmo.log", level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)


def comp_study_single(
    asmo: ASMO,
    plot_results: bool,
    # logger: logging.Logger | None,
    csv_out: str,
    strategy_subproblem_selection: str,
    strategy_node_priority: str,
    strategy_reduction: str | int,
    strategy_update_node_priority: str,
    strategy_refine_with_integer_gap: bool,
    strategy_refine_method: str = "fast",
    strategy_search_method: str = "phase1",
    reset_csv: bool = False,
    time_limit=60 * 60 * 3,  # 60 minutes -> 3 hours
    preload_solutions=False,
):
    start_time = perf_counter()
    # if not logger:  # print to stout instead
    #     # logger = logging.getLogger("comp_study_single")
    #     # logger.addHandler(logging.NullHandler())
    #     logger = logging.getLogger()
    #     logger.addHandler(logging.StreamHandler())
    #
    asmo.set_strategy_config(
        strategy_subproblem_selection=strategy_subproblem_selection,
        strategy_node_priority=strategy_node_priority,
        strategy_reduction=strategy_reduction,
        strategy_update_node_priority=strategy_update_node_priority,
        strategy_refine_with_integer_gap=strategy_refine_with_integer_gap,
        strategy_refine_method=strategy_refine_method,
        strategy_search_method=strategy_search_method,
    )

    logger.info("Starting ASMO solve")
    logger.info(f"\tASMO configuration")
    logger.info(f"\t\tInstance name: {asmo.statistics['filename']}")
    logger.info(f"\t\tstrategy_subproblem_selection: {strategy_subproblem_selection}")
    logger.info(f"\t\tstrategy_node_priority: {strategy_node_priority}")
    logger.info(f"\t\tstrategy_reduction: {strategy_reduction}")
    logger.info(f"\t\tstrategy_update_node_priority: {strategy_update_node_priority}")
    logger.info(f"\t\tstrategy_refine_method: {strategy_refine_method}")
    logger.info(
        f"\t\tstrategy_refine_with_integer_gap: {strategy_refine_with_integer_gap}"
    )
    asmo.print_strategy_config()
    # set strategies
    asmo.get_reduced_generator()
    if True:
        asmo.get_minimum_generator()
    else:
        for s, p in enumerate(asmo.P):
            asmo.Y_mgs = asmo.reduced_generator[s]

    if True:
        for p in asmo.P:
            p.search_method = "random"
            p.search_method = "bbm"
            p.search_method = "phase1"
            p.search_method = asmo.strategy_search_method

    # asmo.Yn = reduce(lambda hY1, hY2: (hY1 + hY2).N(), [p.Yn for p in asmo.P])
    if sum(len(p.Yn) for p in asmo.P) > 0:
        asmo.Yn = reduce(lambda hY1, hY2: hY1.ND_sum(hY2), [p.Yn for p in asmo.P])
    else:
        asmo.Yn = SolutionList([])

    if plot_results:
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

        plotter.save("tests/test_asmo_solver/test_asmo_solver.pdf")

    if plot_results:  # plot algorithm
        success = asmo.solve(plotter)
    else:
        try:
            asmo.solve(None, time_limit=time_limit)
            success = True
        except TimeoutError:
            logger.info("\t\t*** TIMEOUT ***")
            # write statistics to file (add header if file is empty)
            with open(csv_out, "a", newline="") as csvfile:
                fieldnames = list(asmo.statistics.keys())
                if csvfile.tell() == 0:
                    csvfile.write(",".join(fieldnames) + "\n")
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(asmo.statistics)
            success = False
        except AttributeError as e:
            logger.error(f"AttributeError during solve: {e}")
            success = False
    solve_time = perf_counter() - start_time if success else 99999999
    print_timeit()
    # print_timeit()
    # hYn = ((asmo.P[0].hY + asmo.P[1].hY).N() + asmo.P[2].hY).N()
    # hYn = reduce(lambda x, y: (x + y).N(), [p.hY for p in asmo.P])

    if True:
        for p in asmo.P:
            p.hY = p.hY.N()
            p.Yn = p.Yn.N()
        asmo.hY = asmo.hY.N()

    hYn = asmo.hY
    print(len(hYn))
    Yn = reduce(lambda x, y: (x + y).N(), [p.Yn for p in asmo.P])
    print(len(Yn))

    if plot_results:
        plotter.plot(Yn, ax="main", color="red", name=f"|Yn|={len(Yn)}", s=6)
        plotter.plot(hYn, ax="main", color="blue", name=f"|hY|={len(hYn)}", s=3)

    subproblem_points = [len(p.Yn) for p in asmo.P]
    subproblem_points_generated = [len(p.hY) for p in asmo.P]
    reduced_generator_points = [len(_Y) for _Y in asmo.reduced_generator]
    mgs_points = [len(p.Y_mgs) for p in asmo.P]

    print(f"Subproblem points: {subproblem_points}, Total: {sum(subproblem_points)}")
    print(
        f"(Reduced ) Generator points: {reduced_generator_points}, Total: {sum(reduced_generator_points)}"
    )
    print(
        f"Subproblem points generated: {subproblem_points_generated}, Total: {sum(subproblem_points_generated)}"
    )
    print(f"MGS points: {mgs_points}, Total: {sum(mgs_points)}")

    # add three above plots to title
    if plot_results:
        plotter.axs["main"].set_title(
            f"Subproblem points: {subproblem_points}\n Generated: {subproblem_points_generated}\n MGS: {mgs_points}"
        )

        plotter.save("tests/test_asmo_solver/test_asmo_solver.pdf")

    print(len(Yn))
    print(len(hYn))

    print_timeit()

    if success:  # dont run if timeout
        if not len(hYn) == len(Yn):
            print(f"Error: hYn and Yn have different lengths: {len(hYn)} vs {len(Yn)}")
            logger.error(
                f"Error: hYn and Yn have different lengths: {len(hYn)} vs {len(Yn)}"
            )
            # raise ValueError("hYn and Yn have different lengths")

        # assert len(hYn) == len(Yn)
        print(len(hYn) == len(Yn))
        for y in Yn:
            pass
            # print(f"Checking point {y} in Yn is in hYn...")
            # print(AsmoPoint(y.val) in hYn.as_pointlist())
            # assert AsmoPoint(y.val) in hYn.as_pointlist(), f"Point {y} in Yn not in hYn"

    # add strategy
    asmo.statistics.update(asmo.get_strategy_dict())
    # add statistics
    asmo.statistics["time_total"] = solve_time
    asmo.statistics["subproblem_points"] = sum(subproblem_points)
    asmo.statistics["reduced_generator_points"] = sum(reduced_generator_points)
    asmo.statistics["subproblem_points_generated"] = sum(subproblem_points_generated)
    asmo.statistics["mgs_points"] = sum(mgs_points)
    asmo.statistics["Yn_points"] = len(Yn)
    asmo.statistics["hYn_points"] = len(hYn)
    # add vectors
    as_vector = lambda l: "|".join((str(i) for i in l))
    asmo.statistics["subproblem_points_vector"] = as_vector(subproblem_points)
    asmo.statistics["subproblem_points_generated_vector"] = as_vector(
        subproblem_points_generated
    )
    asmo.statistics["mgs_points_vector"] = as_vector(mgs_points)
    asmo.statistics["generator_points_vector"] = as_vector(reduced_generator_points)
    # add timing of selected methods
    selected_method_names = [
        "IP-calls",
        "nodes explored",
        "subproblem_points_vector",
        "subproblem_points_generated_vector",
        "mgs_points_vector",
        "ASMO.solve",
        "ASMO._set_global_upper_bound",
        "ASMO._update_global_upper_bound",
        "ASMO._set_incumbent",
        "CVRP._update_bounds_from_node",
        "ASMO._reduce_upper_bounds",
    ]
    for method in selected_method_names:
        asmo.statistics[f"time_{method}"] = TIME_dict.get(method, None)
        asmo.statistics[f"count_{method}"] = COUNT_dict.get(method, None)
        # if not none round to 3 decimals
        if asmo.statistics[f"time_{method}"] is not None:
            asmo.statistics[f"time_{method}"] = round(
                asmo.statistics[f"time_{method}"], 3
            )

    asmo.statistics["timeout"] = not success

    if not success:
        logger.info(f"ASMO solve failed for instance {asmo.statistics['filename']}")

    logger.info("\tASMO statistics:")
    for k, v in asmo.statistics.items():
        if k in selected_method_names or k in [
            "time_ASMO._reduce_upper_bounds",
            "count_ASMO._reduce_upper_bounds",
        ]:
            logger.info(f"\t\t{k}: {v}")
    # reset csv file if flag activated
    if reset_csv:
        open(csv_out, "w").close()

    # write statistics to file (add header if file is empty)
    with open(csv_out, "a", newline="") as csvfile:
        fieldnames = list(asmo.statistics.keys())
        if csvfile.tell() == 0:
            csvfile.write(",".join(fieldnames) + "\n")
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(asmo.statistics)
    print_timeit()

    return asmo


def file_to_asmo(filename):

    subproblem_path = "./instances/subproblems_local_sets/"
    asmo = ASMO()
    content = json.load(open(filename, "r"))[0]

    print(content)
    for s, sp_filename in content.items():
        sp_name = sp_filename.replace("subproblems/", "")
        Y = PointList.from_json(subproblem_path + sp_name)
        p = CVRP("CVRP_neg_0_16_8_35_NEW.lp")
        p.preload_solutions()
        p.name = sp_name
        X = SolutionList.from_pointlist_dummy(Y, s)
        p.Yn = X
        p.Yse = X.get_supported()
        asmo.load_subproblem(p)

    # asmo = ASMO.from_msp_file(filename)
    asmo.statistics["filename"] = filename

    return asmo


def instance_name_dict(problem_file):
    problem_file = problem_file.split("Lyngesen24-")[-1]
    filename = problem_file
    problem_file = problem_file.split(".json")[0]
    problem_file, seed = problem_file.split("_")
    # print(f"{problem_file=}")
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


if __name__ == "__main__":
    pass

    SMALL_TEST = True

    ASMO = time_object(ASMO)
    CVRP = time_object(CVRP)
    PointList = time_object(PointList)

    asmo_filenames = [
        "./instances/msp/2obj/Lyngesen24-msp-2-100|100-ul-2_1.json",
        # "./instances/msp/2obj/Lyngesen24-msp-2-50|50-ul-2_1.json",
        # "./instances/msp/2obj/Lyngesen24-msp-2-50|50-ul-2_2.json",
        # "./instances/msp/2obj/Lyngesen24-msp-2-50|50-ul-2_3.json",
        "./instances/msp/2obj/Lyngesen24-msp-2-50|50-ul-2_4.json",
        # "./instances/msp/2obj/Lyngesen24-msp-2-50|50-mm-2_4.json",
        # "./instances/msp/2obj/Lyngesen24-msp-2-50|50-ul-2_1.json",
        "instances/msp/2obj/Lyngesen24-msp-2-50|50|50-ull-3_1.json",
    ]

    instance_dir = "./instances/msp/2obj/"
    files = os.listdir(instance_dir)

    instance_dict = {
        "max_size": 100,
        "max_subproblems": 3,
        "min_subproblems": 2,
        "max_seed": 20,
    }

    if SMALL_TEST:
        instance_dict["max_size"] = 50
        instance_dict["max_seed"] = 1
        instance_dict["min_subproblems"] = 3

    # filter out if size exceeds 100 and M exceeds 3 . use instance_name_dict
    asmo_filenames = [
        instance_dir + f
        for f in files
        if all(
            (
                instance_name_dict(f)["size"] <= instance_dict["max_size"],  # 200,
                instance_name_dict(f)["M"] <= instance_dict["max_subproblems"],
                instance_name_dict(f)["M"] >= instance_dict["min_subproblems"],
                instance_name_dict(f)["seed"] <= instance_dict["max_seed"],
                not set(instance_name_dict(f)["method"])
                == set("l"),  # exclude exclusive 'l' method
            )
        )
    ]

    # if SMALL_TEST keep only if set(instance_name_dict(f)["method"]) == set("ul") or set(instance_name_dict(f)["method"]) == set("ull") for f in asmo_filenames

    if SMALL_TEST:
        print(f"{len(asmo_filenames)=}")
        asmo_filenames = [
            f
            for f in asmo_filenames
            if set(instance_name_dict(f)["method"]) == set("ul")
        ]
        print(f"{len(asmo_filenames)=}")

    # parse args
    parser = argparse.ArgumentParser(description="Run ASMO computational study")
    parser.add_argument(
        "--filename", help="Run a single instance with the given filename"
    )
    csv_out_dir = "results/data/"
    parser.add_argument("--csv_out", help="output csv file", default="test.csv")
    # set reset_csv to true if --reset is passed, false otherwise
    parser.add_argument(
        "--reset", action="store_true", help="Reset the csv file before running"
    )
    # filename = args.filename if args.filename else None
    args = parser.parse_args()
    if args.filename:
        print(f" Filename provided, testing only instance {args.filename=}")
        asmo_filenames = [instance_dir + args.filename]
    # reset_csv = args.reset_csv

    # print(f"{asmo_filenames=}")

    # asmo = file_to_asmo("./instances/msp/2obj/Lyngesen24-msp-2-50|50-ul-2_1.json")
    # asmo = file_to_asmo("./instances/msp/2obj/Lyngesen24-msp-2-100|100-ul-2_1.json")
    csv_out = "results/data/test.csv"
    csv_out = "results/data/test_march.csv"
    csv_out = "results/data/test_april.csv"
    csv_out = "results/data/test_may.csv"
    csv_out = csv_out_dir + args.csv_out
    json_out = csv_out.replace(".csv", ".json")

    # if reset or csv_out does not exists
    if args.reset or not os.path.exists(csv_out):  # True to reset
        print(f"Resetting csv file: {csv_out} {args.reset,os.path.exists(csv_out)=}")
        open(csv_out, "w").close()

    config_dict = {
        "strategy_subproblem_selection": (
            "alternating",
            "sequential",
        ),  # TODO:One should also sort the subproblems such that l is solved before u.
        "strategy_node_priority": ("largest",),
        # "strategy_reduction": ("first", 100),
        # "strategy_reduction": ("first", 100),
        "strategy_reduction": ("first", "none", 100, 50),
        # "strategy_reduction": (50, 100),
        # "strategy_reduction": (100,),
        "strategy_update_node_priority": ("all",),
        "strategy_refine_with_integer_gap": (
            False,
            # True,
        ),
        "strategy_refine_method": ("slow",),
        # "strategy_refine_method": ("fast",),
        "strategy_search_method": (
            "bbm",
            "phase1",
        ),
    }

    if True:  # final
        config_dict = {
            "strategy_subproblem_selection": (
                "sequential",
                "alternating",
            ),  # TODO:One should also sort the subproblems such that l is solved before u.
            "strategy_node_priority": ("largest",),
            # "strategy_reduction": ("first", 100),
            # "strategy_reduction": ("first", 100),
            "strategy_reduction": ("first", "none", 100, 50),
            # "strategy_reduction": (50, 100),
            # "strategy_reduction": (100,),
            "strategy_update_node_priority": ("all",),
            "strategy_refine_with_integer_gap": (
                False,
                # True,
            ),
            "strategy_refine_method": ("slow",),
            # "strategy_refine_method": ("fast",),
            "strategy_search_method": (
                "bbm",
                "phase1",
            ),
        }
    if SMALL_TEST:  # small-test
        config_dict = {
            "strategy_subproblem_selection": (
                "sequential",
                "alternating",
            ),  #
            "strategy_node_priority": ("largest",),
            "strategy_reduction": ("first", "none", 50),
            "strategy_update_node_priority": ("all",),
            "strategy_refine_with_integer_gap": (False,),
            "strategy_refine_method": ("slow",),
            "strategy_search_method": (
                "bbm",
                "phase1",
            ),
        }

    if not args.reset and os.path.exists(json_out):
        print(f"Loading config_dict from json file: {json_out}")
        with open(json_out, "r") as f:
            config_dict_load = json.load(f)

        if not all(
            set(config_dict_load[k]) == set(config_dict[k]) for k in config_dict.keys()
        ):

            print(
                f"Config dict values do not match, using loaded config_dict: {config_dict_load}"
            )
            logger.warning(
                f"Config dict values do not match, using loaded config_dict: {config_dict_load}"
            )

            raise ValueError(
                "Config dict values do not match the saved configuration. "
                "Use --reset to overwrite existing results."
            )

    # save config_dict to json file
    with open(json_out, "w") as f:
        out_dict = instance_dict.copy()
        out_dict.update(config_dict)
        json.dump(out_dict, f, indent=4)

    total_runs = 1
    for v in config_dict.values():
        total_runs *= len(v)
    print(f"Total runs per instance: {total_runs}")

    # get solved instances

    with open(csv_out, "r") as f:
        reader = csv.DictReader(f)
        solved_instances = set()
        for row in reader:
            solved_instances.add(
                (
                    row["filename"],
                    row["strategy_search_method"],
                    row["strategy_subproblem_selection"],
                    row["strategy_node_priority"],
                    row["strategy_reduction"],
                    row["strategy_update_node_priority"],
                    row["strategy_refine_with_integer_gap"],
                    row["strategy_refine_method"],
                )
            )

    # import builtins
    # original_print = builtins.print
    # builtins.print = tqdm.write

    for (
        asmo_name,
        strategy_subproblem_selection,
        strategy_node_priority,
        strategy_update_node_priority,
        strategy_refine_with_integer_gap,
        strategy_refine_method,
        strategy_search_method,
        strategy_reduction,
    ) in tqdm(
        product(
            asmo_filenames,
            config_dict["strategy_subproblem_selection"],
            config_dict["strategy_node_priority"],
            config_dict["strategy_update_node_priority"],
            config_dict["strategy_refine_with_integer_gap"],
            config_dict["strategy_refine_method"],
            config_dict["strategy_search_method"],
            config_dict["strategy_reduction"],
        ),
        total=len(asmo_filenames) * total_runs,
        initial=len(solved_instances),
    ):

        reset_timeit()
        from src.asmo.utils.timing import TIME_dict, COUNT_dict

        asmo = file_to_asmo(asmo_name)

        # check if solved
        if (
            asmo.statistics["filename"],
            strategy_search_method,
            strategy_subproblem_selection,
            strategy_node_priority,
            str(strategy_reduction),
            strategy_update_node_priority,
            str(strategy_refine_with_integer_gap),
            str(strategy_refine_method),
        ) in solved_instances:
            print(
                f"Skipping already solved instance: {asmo.statistics['filename']}, {strategy_subproblem_selection}, {strategy_node_priority}, {strategy_reduction}, {strategy_update_node_priority}, {strategy_refine_with_integer_gap}"
            )
            continue

        comp_study_single(
            asmo=asmo,
            plot_results=False,
            csv_out=csv_out,
            strategy_subproblem_selection=strategy_subproblem_selection,
            strategy_node_priority=strategy_node_priority,
            strategy_reduction=strategy_reduction,
            strategy_update_node_priority=strategy_update_node_priority,
            strategy_refine_with_integer_gap=strategy_refine_with_integer_gap,
            strategy_refine_method=strategy_refine_method,
            strategy_search_method=strategy_search_method,
            reset_csv=False,
        )

        print_timeit()
    print(f"results saved in {csv_out=}")
