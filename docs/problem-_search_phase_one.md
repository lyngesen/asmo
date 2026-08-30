---
id: problem-_search_phase_one
aliases: []
tags: []
---

def _search_phase_one(self, n: BnbNode):
    search_phase_one. Search the node n using phase one method

This methods applies the phase one method for finding all supported points inside the branching node n, which might be unsupported in the subproblem.
    This is done by adding a constraint on each objective $f1(x) <= R1$ and $f2(x) <= R2$ (restricting solutions to be inside the box).
    After a set of solutions is found (potentially none) the lower and upper bound sets are constructed/updated using these solutions.

Flags:
        self.solutions_preloaded: If True, the method uses preloaded solutions from self.Yn to define the set of solutions inside the branching node n. Otherwise, a TODO: implement actual search using IP solver

Args:
        n: a search node

Raises:
        NotImplementedError: [TODO:description]


# Explanation

## Finding new solutions

The search method assumes the following are defined for the problem $P$ and search node $\eta$:
- Problem: A MIP model or a set of preloaded ND-points $\mathcal{Y}_N$
- Search node: A subproblem $s(\eta) = s$, A rectangle $R^\eta$ defined by the points $\eta^{ul}, \eta^{lr},\eta^{ll}$ and $\eta^{ur}$.


If solutions are preloaded — $\mathcal{Y}_N^{s}$ is given —, then the algorithm selects all extreme supported points in $\tilde{\mathcal{Y}}^\eta \coloneq \left(\mathcal{Y}_N^s \cap R^\eta\right)_{SE}$. Otherwise, if the solutions are not preloaded — in which case a MIP model is defined — the two constraints are added to the MIP model, constraining the objective vectors to be in the rectangle $R^\eta$:
$$f^s(x)_1 \le \eta^{lr}_1$$
$$f^s(x)_2 \le \eta^{ul}_2$$
The set $\tilde{\mathcal{Y}}^\eta$ is then obtained by running the phase one method, which the added constraints.

## Updating bounds
### If no new solutions are found
If no new solutions are found the lower and upper bound sets of the search node $\eta$ will be defined from the upper and lower part of the rectangle, respectively.
$$\mathcal{U}^\eta = LineString(\eta^{ul}, \eta^{ll} , \eta^{lr})$$
$$\mathcal{L}^\eta = LineString(\eta^{ul}, \eta^{ul} , \eta^{lr})$$
For example see last figure first node (upper left node).
### If new solutions are found

If new solutions are found by running the phase one method we use these to define the bound sets of the node $\eta$.
We sort the set $\tilde{\mathcal{Y}}^\eta$ (lexicographical) and let $\tilde{y}^{ul}$ and $\tilde{y}^{lr}$ denote the leftmost and rightmost point of $\tilde{\mathcal{Y}}^\eta$
#### Upper bound

The points $\tilde{\mathcal{Y}}^\eta$ would — as incumbent solutions — constitute an upper bound set for $\mathcal{Y}_N^{s} \cap R^\eta$. Additionally, the set $\left\{ \pmatrix{\eta^{ul}_1\\ \tilde{y}^{ul}_2}  , \pmatrix{\eta^{lr}_1\\ \tilde{y}^{lr}_2} \right\} \cup \tilde{\mathcal{Y}}^\eta$ constitute and upper bound set for the node $\eta$. In our implementation using LineStrings we define the upper bound set as follows:
$$\mathcal{U}^\eta \coloneqq  LineString\left( \left\{ \pmatrix{\eta^{ul}_1\\ \tilde{y}^{ul}_2}\right\} \cup LocalNadirPoints\left(\tilde{\mathcal{Y}}^\eta\right)  \cup \left\{\pmatrix{\eta^{lr}_1\\ \tilde{y}^{lr}_2} \right\}\right)$$
#### Lower bound

Since the set $\tilde{\mathcal{Y}}^\eta = \left(\mathcal{Y}_N^s \cap R^\eta\right)_{SE}$ contains all supported efficient points of the set $\mathcal{Y}_N^s \cap R^\eta$, we have that $(\text{conv}(\tilde{\mathcal{Y}}^\eta))_N$ constitute a lower bound for $\mathcal{Y}_N^{s}\cap R^\eta$. Expressed as a LineString from $\eta^{ul}$ to $\eta^{lr}$ we define the lower bound set as:
$$\mathcal{L}^\eta \coloneqq  LineString\left( \left\{ \pmatrix{ \tilde{y}^{ul}_1 \\\eta^{ul}_2}\right\} \cup \text{sorted}\left(\tilde{\mathcal{Y}}^\eta\right)  \cup \left\{\pmatrix{ \tilde{y}^{lr}_1 \\ \eta^{lr}_2} \right\}\right)$$

## Visualization(s)

Created in tests/test_problem.py :test_update_bounds_from_node

![Visualisation of updates using found solutions](tests/test_update_bounds_from_node_points.pdf)

Visualisation of several updates:
- Note how no points in the upper left node is found: hence the bound sets are defines as described in [[problem-_search_phase_one#updating-bounds#if-no-new-solutions-are-found|Explanation ❯ If no new solutions are found]] 

![Visualisation of several updates](tests/test_update_bounds_from_node_all.pdf) 
