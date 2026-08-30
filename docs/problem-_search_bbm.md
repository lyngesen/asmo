---
id: problem-_search_bbm
aliases: []
tags: []
---
```search the node using the balanced-box mehtod (bbm)
Args:
        n (BnbNode): node to be searched
Modifies:
        n.tY (SolutionList): set of solutions returned by search
        n.tL (Bound): lower bound after search
        n.tU (Bound): upper bound after search
        n.T (BranchingTree): branching tree associated with the node
```

# Explanation

This search method is based on the `Balanced Box method` (see [[boland15]]).

The algorithm performs the following steps:

1. Find new solutions $y^1$ and $y^2$ by solving lex-max problems.
2. Create new branching nodes $n^1$ and $n^2$ defined by rectangles $R^1$ and $R^2$.
3. Create upper and lower bound sets for generator points inside the area $R^\eta$, saved as `n.tU` and `n.tL`, respectively.

## Finding new solutions


The search method assumes the following are defined for the problem $P$ and search node $\eta$:
- Problem: A MIP model or a set of preloaded ND-points $\mathcal{Y}_N$
- Search node: A subproblem $s(\eta) = s$, A rectangle $R^\eta$ defined by the points $\eta^{ul}, \eta^{lr},\eta^{ll}$ and $\eta^{ur}$.



First a box $R^B$ is defined containing the lower half of $R^\eta$.
Let $c= \frac{ul(R^\eta)_2 + lr(R^\eta)_2}{2}$ be the center value of objective $2$.
$$R^B = Rect[\pmatrix{ul(R^\eta)_1\\ c}, lr(R^\eta)]$$

$y^1 = \arg \max_{y \in R^B \cup \mathcal{Y}_N^{s} } (y_1 + M y_2)$

if 


$y^2 = arg lex max ()$

The set $\tilde{\mathcal{Y}}^\eta$ is then obtained by running the balanced box method on the node, which the added constraints.

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
