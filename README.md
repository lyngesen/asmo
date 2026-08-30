# Additively-separabel multi-objective optimization Implementation

Implementation for paper on additively-separable multi-objective optimization problems.

## Project structure

The repository is organized as follows:

```text
.
├── instances/               # Input instances (cvrp, msp, subproblems_local_sets, tests)
├── results/                 # Generated CSV files, plots, and data
├── src/asmo/                # Main Python package
│   ├── classes/             # Problem instances, solutions, and optimization models
│   │   ├── asmo.py
│   │   ├── geom.py
│   │   ├── plotter.py
│   │   ├── pointsets.py
│   │   └── problem.py
│   └── utils/               # Helper functions, generators, and algorithms
│       ├── fast_A_dominated_by_B.py
│       ├── fast_bound_ms.py
│       ├── fastMinimumGenerator.py
│       ├── minimumGenerator.py
│       ├── mspMethods.py
│       ├── shapely_get_corner_points.py
│       ├── timing.py
│       └── upper_envelope.py
├── tests/                   # Unit tests
├── comp_study.py            # Runs computational experiments
├── requirements.txt         # Python dependencies
├── LICENSE                  # License file
├── nondom                   # ND filter (C library)
├── ND_pointsSum2            # ND filter for Minkowski sums (C library)
└── README.md                # This file
```

The main implementation is contained in `src/asmo`, while experiment scripts, input instances, and generated results are kept in separate top-level directories.

## Installation

To install and run this project, follow these steps:

1. **Clone the repository** (if not already done)

2. **Setup virtual environment** (assuming Python 3 is installed):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

4. **Run computational studies**:
   ```bash
   python comp_study.py
   ```

## Usage

Run the main computational study with:

```bash
python comp_study.py --csv-out=<name of csv file>
```

This will generate results CSV files that can be found in the `results/` directory.

## Instances

Test instances are provided in the `instances/` directory, including:
- `msp/` - Multi-objective Set Packing Problem instances. Taken from [MOrepo-Lyngesen24](https://github.com/lyngesen/MOrepo-Lyngesen24).
- `subproblems_local_sets/` - Subproblem instances. Taken from [MOrepo-Lyngesen24](https://github.com/lyngesen/MOrepo-Lyngesen24).
- `cvrp/` - A selection of bi-objective mTSP instances.
