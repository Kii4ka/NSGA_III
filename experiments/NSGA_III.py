#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/22 09:14
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : NSGA_III.py
# @Statement : Nondominated sorting genetic algorithm III (NSGA-III)
# @Reference : K. Deb and H. Jain, An evolutionary many-objective optimization algorithm using reference-point based non-dominated sorting approach, part I: Solving problems with box constraints, IEEE Transactions on Evolutionary Computation, 2014, 18(4): 577-601.
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from scipy.linalg import LinAlgError
from scipy.spatial.distance import cdist
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def cal_obj(pop, nobj):
    # DTLZ1
    g = 100 * (pop.shape[1] - nobj + 1 + np.sum((pop[:, nobj - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (pop[:, nobj - 1:] - 0.5)), axis=1))
    objs = np.zeros((pop.shape[0], nobj))
    temp_pop = pop[:, : nobj - 1]
    for i in range(nobj):
        f = 0.5 * (1 + g)
        f *= np.prod(temp_pop[:, : temp_pop.shape[1] - i], axis=1)
        if i > 0:
            f *= 1 - temp_pop[:, temp_pop.shape[1] - i]
        objs[:, i] = f
    return objs


def factorial(n):
    # calculate n!
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def combination(n, m):
    # choose m elements from an n-length set
    if m == 0 or m == n:
        return 1
    elif m > n:
        return 0
    else:
        return factorial(n) // (factorial(m) * factorial(n - m))


def reference_points(npop, nvar):
    # calculate approximately npop uniformly distributed reference points on nvar dimensions
    h1 = 0
    while combination(h1 + nvar, nvar - 1) <= npop:
        h1 += 1
    points = np.array(list(combinations(np.arange(1, h1 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
    points = (np.concatenate((points, np.zeros((points.shape[0], 1)) + h1), axis=1) - np.concatenate((np.zeros((points.shape[0], 1)), points), axis=1)) / h1
    if h1 < nvar:
        h2 = 0
        while combination(h1 + nvar - 1, nvar - 1) + combination(h2 + nvar, nvar - 1) <= npop:
            h2 += 1
        if h2 > 0:
            temp_points = np.array(list(combinations(np.arange(1, h2 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
            temp_points = (np.concatenate((temp_points, np.zeros((temp_points.shape[0], 1)) + h2), axis=1) - np.concatenate((np.zeros((temp_points.shape[0], 1)), temp_points), axis=1)) / h2
            temp_points = temp_points / 2 + 1 / (2 * nvar)
            points = np.concatenate((points, temp_points), axis=0)
    return points


def nd_sort(objs):
    # fast non-domination sort
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)  # the number of individuals that dominate this individual
    s = []  # the index of individuals that dominated by this individual
    rank = np.zeros(npop, dtype=int)
    ind = 0
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind
    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1
    pfs.pop(ind)
    return pfs, rank


def selection(pop, pc, rank, k=2):
    # binary tournament selection
    (npop, nvar) = pop.shape
    nm = int(npop * pc)
    nm = nm if nm % 2 == 0 else nm + 1
    mating_pool = np.zeros((nm, nvar))
    for i in range(nm):
        [ind1, ind2] = np.random.choice(npop, k, replace=False)
        if rank[ind1] <= rank[ind2]:
            mating_pool[i] = pop[ind1]
        else:
            mating_pool[i] = pop[ind2]
    return mating_pool


def crossover(mating_pool, lb, ub, pc, eta_c):
    # simulated binary crossover (SBX)
    (noff, nvar) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, nvar))
    mu = np.random.random((nm, nvar))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, (nm, nvar))
    beta[np.random.random((nm, nvar)) < 0.5] = 1
    beta[np.tile(np.random.random((nm, 1)) > pc, (1, nvar))] = 1
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, pm, eta_m):
    # polynomial mutation
    (npop, nvar) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, nvar)) < pm / nvar
    mu = np.random.random((npop, nvar))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def environmental_selection(pop, objs, zmin, npop, V):
    # NSGA-III environmental selection
    pfs, rank = nd_sort(objs)
    nobj = objs.shape[1]
    selected = np.full(pop.shape[0], False)
    ind = 0
    while np.sum(selected) + len(pfs[ind]) <= npop:
        selected[pfs[ind]] = True
        ind += 1
    K = npop - np.sum(selected)

    # select the remaining K solutions
    objs1 = objs[selected]
    objs2 = objs[pfs[ind]]
    npop1 = objs1.shape[0]
    npop2 = objs2.shape[0]
    nv = V.shape[0]
    temp_objs = np.concatenate((objs1, objs2), axis=0)
    t_objs = temp_objs - zmin

    # extreme points
    extreme = np.zeros(nobj)
    w = 1e-6 + np.eye(nobj)
    for i in range(nobj):
        extreme[i] = np.argmin(np.max(t_objs / w[i], axis=1))

    # intercepts
    try:
        hyperplane = np.matmul(np.linalg.inv(t_objs[extreme.astype(int)]), np.ones((nobj, 1)))
        if np.any(hyperplane == 0):
            a = np.max(t_objs, axis=0)
        else:
            a = 1 / hyperplane
    except LinAlgError:
        a = np.max(t_objs, axis=0)
    t_objs /= a.reshape(1, nobj)

    # association
    cosine = 1 - cdist(t_objs, V, 'cosine')
    distance = np.sqrt(np.sum(t_objs ** 2, axis=1).reshape(npop1 + npop2, 1)) * np.sqrt(1 - cosine ** 2)
    dis = np.min(distance, axis=1)
    association = np.argmin(distance, axis=1)
    temp_rho = dict(Counter(association[: npop1]))
    rho = np.zeros(nv)
    for key in temp_rho.keys():
        rho[key] = temp_rho[key]

    # selection
    choose = np.full(npop2, False)
    v_choose = np.full(nv, True)
    while np.sum(choose) < K:
        temp = np.where(v_choose)[0]
        jmin = np.where(rho[temp] == np.min(rho[temp]))[0]
        j = temp[np.random.choice(jmin)]
        I = np.where(np.bitwise_and(~choose, association[npop1:] == j))[0]
        if I.size > 0:
            if rho[j] == 0:
                s = np.argmin(dis[npop1 + I])
            else:
                s = np.random.randint(I.size)
            choose[I[s]] = True
            rho[j] += 1
        else:
            v_choose[j] = False
    selected[np.array(pfs[ind])[choose]] = True
    return pop[selected], objs[selected], rank[selected]


def nsga3_score(B, R, iteration=None, max_label=None):
    """
    Enhanced NSGA-III based scoring function for active learning
    
    B: Best value (first objective) - exploitation component
    R: Rest/uncertainty value (second objective) - exploration component
    iteration: Current iteration in active learning (will be inferred if None)
    max_label: Maximum label budget (will use default scaling if None)
    
    Returns a score based on NSGA-III algorithm, higher is better
    """
    # Handle edge cases
    if np.isnan(B) or np.isnan(R) or (abs(B) < 1e-10 and abs(R) < 1e-10):
        return 0
    
    # Infer current phase of active learning if not provided explicitly
    # This allows the function to adapt to the sampling process
    if iteration is None and max_label is None:
        # Simple heuristic based on B/R values to estimate progress
        # Higher values typically occur later in the sampling process
        total = abs(B) + abs(R)
        if total > 4.0:
            progress = 0.8  # Late stage - favor exploitation
        elif total > 2.0:
            progress = 0.5  # Mid stage - balanced approach
        else:
            progress = 0.2  # Early stage - favor exploration
    elif iteration is not None and max_label is not None:
        # Calculate progress ratio if iteration and max_label are provided
        progress = min(1.0, iteration / max_label)
    else:
        # Default balanced progress
        progress = 0.5
    
    # Dynamic population size that scales with expected complexity
    # Larger populations allow more nuanced comparisons but are computationally expensive
    if progress < 0.3:
        population_size = 12  # Early stage - smaller population is sufficient
    elif progress < 0.7:
        population_size = 16  # Mid stage - medium population
    else:
        population_size = 20  # Late stage - larger population for precision
    
    # Create a diverse population with meaningful trade-offs
    objs = np.zeros((population_size, 2))
    
    # Our candidate point is always the first one
    objs[0] = [B, R]
    
    # Generate diverse reference points for comparison
    # Different strategies based on where we are in the active learning process
    if progress < 0.4:  # Early stage - focus on exploration
        for i in range(1, population_size):
            if i < population_size // 3:
                # Some points with better exploration (R)
                objs[i] = [B * (1.1 + 0.1*i), R * (0.8 - 0.05*i)]
            elif i < 2 * (population_size // 3):
                # Some points with better exploitation (B)
                objs[i] = [B * (0.8 - 0.05*(i-population_size//3)), R * (1.1 + 0.1*(i-population_size//3))]
            else:
                # Some clearly dominated points
                objs[i] = [B * 1.2, R * 1.2]
    else:  # Later stage - more focus on exploitation
        for i in range(1, population_size):
            if i < population_size // 3:
                # Better exploitation points
                objs[i] = [B * (0.7 - 0.05*i), R * (1.0 + 0.1*i)]
            elif i < 2 * (population_size // 3):
                # Trade-off points
                if i % 2 == 0:
                    objs[i] = [B * 0.9, R * 1.1]  # Better B, worse R
                else:
                    objs[i] = [B * 1.1, R * 0.9]  # Worse B, better R
            else:
                # Dominated points
                objs[i] = [B * (1.2 + 0.05*(i-2*population_size//3)), 
                           R * (1.2 + 0.05*(i-2*population_size//3))]
    
    # Generate reference vectors - using standard NSGA-III approach
    V = reference_points(population_size, 2)
    
    # Calculate ideal point
    zmin = np.min(objs, axis=0)
    
    # Get Pareto fronts and ranks
    pfs, rank = nd_sort(objs)
    
    # Find our point's rank
    point_rank = rank[0]
    
    # Create dummy population (required for environmental selection)
    pop = np.zeros((population_size, 2))
    
    try:
        # Normalize objectives for better numerical stability
        t_objs = objs - zmin
        
        # Handle extreme points and intercepts calculation
        nobj = 2
        extreme = np.zeros(nobj, dtype=int)
        w = 1e-6 + np.eye(nobj)
        
        for i in range(nobj):
            extreme[i] = np.argmin(np.max(t_objs / w[i], axis=1))
        
        # Calculate intercepts with defensive programming
        try:
            hyperplane = np.linalg.solve(t_objs[extreme], np.ones(nobj))
            if np.any(hyperplane <= 0) or np.any(np.isinf(hyperplane)):
                a = np.max(t_objs, axis=0)
            else:
                a = 1 / hyperplane
        except np.linalg.LinAlgError:
            a = np.max(t_objs, axis=0)
        
        # Guard against division by zero
        a = np.maximum(a, 1e-10)
        
        # Normalize objectives
        t_objs = t_objs / a.reshape(1, nobj)
        
        # Calculate association with reference vectors
        cosine = 1 - cdist(t_objs, V, 'cosine')
        norm_t = np.sqrt(np.sum(t_objs**2, axis=1)).reshape(-1, 1)
        distance = norm_t * np.sqrt(np.maximum(0, 1 - cosine**2))  # ensure non-negative
        
        # Get our point's association and distance
        point_dist = distance[0]
        min_dist_idx = np.argmin(point_dist)
        min_dist = point_dist[min_dist_idx]
        
        # Calculate niche count (how many points share this reference vector)
        association = np.argmin(distance, axis=1)
        niche_count = np.sum(association == min_dist_idx)
        
        # Dynamic scoring components
        # 1. Rank-based component (lower rank is better)
        rank_score = 10.0 / (1.0 + point_rank)
        
        # 2. Niche-based component (less crowded niches are better)
        niche_score = 5.0 / (niche_count + 1)
        
        # 3. Distance-based component (closer to reference vector is better)
        dist_score = 3.0 / (min_dist + 1)
        
        # Calculate adaptive weights based on progress
        # Early: favor exploration
        # Late: favor exploitation
        exploit_weight = 0.3 + (0.6 * progress)
        niche_weight = 0.4 - (0.2 * progress)
        dist_weight = 0.3 - (0.1 * progress)
        
        # Combined score with adaptive weights
        score = (exploit_weight * rank_score) + \
                (niche_weight * niche_score) + \
                (dist_weight * dist_score)
                
    except Exception as e:
        # Robust fallback mechanism that's still effective
        
        # Simple Pareto rank-based component
        rank_score = 5.0 / (point_rank + 1)
        
        # Balance component - prefer balanced exploitation/exploration
        norm_sum = abs(B) + abs(R) + 1e-10
        B_norm = abs(B) / norm_sum
        R_norm = abs(R) / norm_sum
        
        # Prefer points with good B/R balance in early stages,
        # prefer points with good B (lower is better) in later stages
        if progress < 0.4:
            # Early stage - reward balance
            balance_score = 5.0 * (1.0 - abs(B_norm - R_norm))
            weight_B = 0.3
            weight_balance = 0.7
        else:
            # Later stage - reward performance
            balance_score = 3.0 * (1.0 - abs(B_norm - R_norm))
            weight_B = 0.6
            weight_balance = 0.4
            
        # Final fallback score
        score = (weight_B * (10.0 / (1.0 + B))) + (weight_balance * balance_score)
    
    return score


def main(npop, iter, lb, ub, nobj=3, pc=1, pm=1, eta_c=30, eta_m=20):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param nobj: the dimension of objective space
    :param pc: crossover probability (default = 1)
    :param pm: mutation probability (default = 1)
    :param eta_c: spread factor distribution index (default = 30)
    :param eta_m: perturbance factor distribution index (default = 20)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = cal_obj(pop, nobj)  # objectives
    V = reference_points(npop, nobj)  # reference vectors
    zmin = np.min(objs, axis=0)  # ideal points
    [pfs, rank] = nd_sort(objs)  # Pareto rank

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 50 == 0:
            print('Iteration: ' + str(t + 1) + ' completed.')

        # Step 2.1. Mating selection + crossover + mutation
        mating_pool = selection(pop, pc, rank)
        off = crossover(mating_pool, lb, ub, pc, eta_c)
        off = mutation(off, lb, ub, pm, eta_m)
        off_objs = cal_obj(off, nobj)

        # Step 2.2. Environmental selection
        zmin = np.min((zmin, np.min(off_objs, axis=0)), axis=0)
        pop, objs, rank = environmental_selection(np.concatenate((pop, off), axis=0), np.concatenate((objs, off_objs), axis=0), zmin, npop, V)

    # Step 3. Sort the results
    pf = objs[rank == 0]
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.view_init(45, 45)
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    z = [o[2] for o in pf]
    ax.scatter(x, y, z, color='red')
    ax.set_xlabel('objective 1')
    ax.set_ylabel('objective 2')
    ax.set_zlabel('objective 3')
    plt.title('The Pareto front of DTLZ1')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(91, 400, np.array([0] * 7), np.array([1] * 7))
