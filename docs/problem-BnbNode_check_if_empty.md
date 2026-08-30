---
id: problem-BnbNode_check_if_empty
aliases: []
tags: []
---



# Documentation for pruning nodes

Location src/classes/problem.py : BnbNode._check_if_empty

Doctext:

```
A test to check if the search area is empty. True if the branching node search area intersected with the search area A(L,U) is empty.
Checks if the branching node is empty by intersecting the search area defined by the branching node with the search area A(L,U).
For documentation see docs/documentation/problem-BnbNode_check_if_empty.md

Args:
	L: Lower bound set of subproblen n.L
	U: Upper bound set of subproblem b.U

returns:
	True if empty, else False
```


# Algorithm from paper

Given lower and upper bound sets $L,U$ we can define a search region as $\mathcal{A}(\mathcal{L},\mathcal{U})$ as
$$\mathcal{A}(\mathcal{L},\mathcal{U}) \coloneqq  (\mathcal{L} \oplus \mathbb{R}^p_\geqq) \setminus (\mathcal{U} \oplus \mathbb{R}^p_>)  $$

The search area of a branching node $\eta$ is defined by the intersection of the subproblem search area $\mathcal{A}(\mathcal{L},\mathcal{U})$ and the branching box $R^\eta$. That is,

$$\mathcal{A}^\eta = \mathcal{A}(\mathcal{L},\mathcal{U}) \cap R^\eta$$
Note: $\mathcal{L,U}$ are the bound sets for the subproblem. 


# Implementation details

The implementation uses the `Shapely`package for representation and manipulation of 2d geometric shapes.
$$\mathcal{A}(\mathcal{L},\mathcal{U}) \coloneqq  (\mathcal{L} \oplus \mathbb{R}^p_\geqq) \setminus (\mathcal{U} \oplus \mathbb{R}^p_>)  $$


**Remark:** Assuming $\mathcal{A}(\mathcal{L}, \mathcal{U})$ is bounded then:

$$\mathcal{A}(\mathcal{L},\mathcal{U}) =  (\mathcal{L} \oplus \mathbb{R}^p_\geqq) \cap (\mathcal{U} \ominus \mathbb{R}^p_\geqq)  $$

The calculation of the above defined search area we first calculate the to areas seperately using the functions:
`L.dominates_space()`and `U.dominated_by_space()` which represent $(\mathcal{L} \oplus \mathbb{R}^p_\geqq)$ and $(\mathcal{U} \oplus \mathbb{R}^p_>)$ respectively. Here a pre-specified value `Z`  is used instead of $\infty$, and the returned object is the wanted object intersected with a hypercube of length `Z`. (This value is saved in the `Space` object as `Space.Z`).


![Visual of dominated/dominates space](results/plots/tests/test_dominated_space.pdf)

(Image 19/11/2025)
![[Pasted image 20251119142318.png]]

Then given the to sets the intersection can be calculated with `Shapely.intersection`


# Visualization

an example is implemented in the test tests/test_problem:test_check_if_empty()

The green area shows the set $\mathcal{A}^\eta$.

![Visual of dominated/dominates space](results/plots/tests/check_if_empty.pdf)

(static image 19/11/25)
![[Pasted image 20251119151004.png]]