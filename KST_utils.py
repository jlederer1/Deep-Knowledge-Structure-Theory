# imports 
import numpy as np
import random
import time
from itertools import product
from matplotlib import pyplot as plt
from tqdm import tqdm



# confere OEIS (Online Encyclopedia of Integer Sequences) https://oeis.org/
num_prosets = {2: 4,    3: 29,   4: 355,  5: 6942, 6: 209527, 7: 9535241, 8:642779354, 9:63260289423}
num_posets =  {2: 3,    3: 19,   4: 219,  5: 4231, 6: 130023, 7: 6129859, 8:431723379, 9:44511042511}
num_spaces = {} # Dedekind numbers of union-closed families
num_states =  {2: 4,    3: 8,    4: 16,   5: 32,   6: 64,     7: 128    , 8:256      , 9:512}

def is_transitive(matrix):
    """Check if the given matrix is transitive.
    :param matrix: numpy array representing adjacency matrix
    """
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if matrix[i, j]:
                for k in range(n):
                    if matrix[j, k] and not matrix[i, k]:
                        return False
    return True

def is_antisymmetric(matrix):
    """Check if the given matrix is antisymmetric.
    :param matrix: numpy array representing adjacency matrix
    """
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] and matrix[j, i]:
                return False
    return True

def get_unique_sublists(list_of_lists):
    """
    Return a list of unique sublists from the given list of lists of tuples representing relations.
    Each sublist is considered unique based on the set of tuples it contains,
    regardless of the order of those tuples.
    
    :param list_of_lists: A list where each element is a list of tuples.
    :return: A list of unique sublists.
    """
    # frozenset for comparison and hashability
    # keep track of the original list with a dictionary
    unique_frozensets = {}
    for sublist in list_of_lists:
        fs = frozenset(sublist)
        if fs not in unique_frozensets:
            unique_frozensets[fs] = sublist
    
    # preserves the order of first occurrence
    unique_list_of_lists = list(unique_frozensets.values())
    
    return unique_list_of_lists

# handling set notation for relations
def transitive(t, Relation) :
    # checks if target relation pair t is transitive with respect to relation R
    R_reduced = [ (i,j) for (i,j) in Relation if i!=j]
    if Relation :
      for e in R_reduced :
        if t[0] == e[1] and (e[0], t[1]) not in Relation :
          return False
        elif e[0] == t[1] and (t[0], e[1]) not in Relation :
          return False
    return True

def R_to_matrix(Relation):
    """
    Translates a Relation (list of item tuples) to a binary np square matrix.
    The size of the matrix is inferred from the number of reflexive pairs in A.
    
    :param: Relation set of item pairs.
    :return: A binary np square matrix.
    """
    items = set()
    for pair in Relation:
        items.update(pair)
    m = len(items)
    matrix = np.zeros((m, m), dtype=int)
    for i in range(m):
        for j in range(m):
            if (j, i) in Relation:
                matrix[j][i] = 1
    
    return matrix

def matrix_to_R(matrix):
    """
    Converts a binary matrix into a set of item pairs (Relation) representing the relationships defined by the matrix.
    
    :param matrix: A binary numpy matrix where a 1 at position (i, j) indicates a relation from item i to item j.
    :return: A set of tuples, each representing a relationship between items.
    """
    Relation = set()
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 1:
                Relation.add((i, j))
                
    return Relation

def construct_relations(D):
    # IITA construction procedure for quasi-order selection set 
    # given a dataset of response patterns as np matrix
    m = D.shape[0]
    B = calculate_counterexamples(D)
    print(m)
    relations = []
    current = []

    # Construct one model (quasi-order) for each tolerance level L
    for L in range(m):
        if relations:
            current = relations[-1]

        # Potential new implications (i,j) in A with associated tolerance level l <= L
        A1 = []     
        for (i,j),_ in np.ndenumerate(B):
            if (B[i,j] < L+1) and ((i,j) not in current):
                A1 += [(i,j)]
        # excludes implications that cause intransitivity to potential new relation (quasi-order)
        A2 = []
        for (i,j) in A1:
            if transitive((i,j), current + A1): 
                A2 += [(i,j)]

        next_r = current + A2
        # store new quasi-order
        if next_r not in relations:
            relations += [next_r]

    return relations

### Source: https://github.com/milansegedinac/kst/blob/master/learning_spaces/kst/imp2state.py
def surmise_matrix_to_states(surmise_matrix):
    """
    Transform a surmise relation matrix to a set of knowledge states (quasi ordinal knowledge space).

    :param surmise_matrix: numpy array representing the surmise relation
    :return: List of knowledge states represented as binary matrices
    """
    items = surmise_matrix.shape[0]

    # Find implications from the surmise matrix
    implications = set()
    for i in range(items):
        for j in range(items):
            if i != j and surmise_matrix[i, j] == 1:
                implications.add((i, j))

    # Transformation from Implications to Knowledge States
    R_2 = np.ones((items, items))
    for i in range(items):
        for j in range(items):
            if (i != j) and ((i, j) not in implications):
                R_2[j, i] = 0

    base = []
    for i in range(items):
        tmp = []
        for j in range(items):
            if R_2[i, j] == 1:
                tmp.append(j)
        base.insert(i, tmp)

    base_list = []
    for i in range(items):
        base_list.insert(i, set())
        for j in range(len(base[i])):
            base_list[i].update(frozenset([base[i][j]]))

    G = []
    G.insert(0, {frozenset()})
    G.insert(1, set())
    for i in range(len(base[0])):
        G[1].update(frozenset([base[0][i]]))
    G[1] = {frozenset(), frozenset(G[1])}

    for i in range(1, items):
        H = {frozenset()}
        for j in G[i]:
            if not base_list[i].issubset(j):
                for d in range(i):
                    if base_list[d].issubset(j.union(base_list[i])):
                        if base_list[d].issubset(j):
                            H.update(frozenset([j.union(base_list[i])]))
                    if not base_list[d].issubset(j.union(base_list[i])):
                        H.update(frozenset([j.union(base_list[i])]))
        G.insert(i+1, G[i].union(H))

    P = np.zeros((len(G[items]), items), dtype=np.int8)
    i = 0
    sorted_g = [list(i) for i in G[items]]
    sorted_g.sort(key=lambda x: (len(x), x))

    for k in sorted_g:
        for j in range(items):
            if j in k:
                P[i, j] = 1
        i += 1

    return P

def sample_with_blim_2(K, num_samples=10, p_k=None, beta=None, eta=None, seed=None):
    """
    Sample response patterns from a knowledge structure using the Binary Latent Item Model (BLIM).

    :param K: Knowledge structure represented as a binary matrix (n x m)
    :param num_samples: Number of response patterns to sample
    :param p_k: Probabilities of knowledge states (default is uniform)
    :param beta: Probabilities of a careless error (default is 0.1 for all items)
    :param eta: Probabilities of a lucky guess (default is 0.1 for all items)
    :param seed: Seed for random number generation (optional)
    :return: List of sampled response patterns represented as binary vectors
    """
    n, m = K.shape
    patterns = []

    if p_k is None:
        p_k = np.ones(n) / n  # Uniform probabilities if not provided

    if beta is None:
        beta = 0.1  # Default beta values
    beta = np.full(m, beta)  

    if eta is None:
        eta = 0.1	# Default eta values
    eta = np.full(m, eta)     

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for _ in range(num_samples):
        pattern = np.zeros(m, dtype=int)
        for i in range(m):
            # Error in original implementation:
            #   adding the probabilities of a correct answer and a lucky guess before multiplying by the state probability can lead to summing up probabilities incorrectly
            #   and results in prob < 0 or prob > 1...
            ### prob = np.sum(K[:, i] * p_k * (1 - beta[i]) + (1 - K[:, i]) * (1 - p_k) * eta[i]) ###
            # Instead:
            prob_correct = np.sum(K[:, i] * p_k * (1 - beta[i]))
            prob_lucky = np.sum((1 - K[:, i]) * p_k * eta[i])
            prob = prob_correct + prob_lucky

            # print(f"DEBUG blim: {prob} = {prob_correct} + {prob_lucky}")
            pattern[i] = np.random.choice([0, 1], p=[1 - prob, prob])
        patterns.append(pattern)

    return patterns

def compute_r(R, K, beta, eta):
    # Computes the probability of the response pattern R given the knowledge state K 
    # Lucky guess probability eta and careless error probability beta are kept constant for all items
    R = R.astype(bool)
    K = K.astype(bool)
    
    LG_response = R & (~K)
    CE_response = K & (~R)
    correct_response = R & K
    incorrect_response = (~K) & (~R) 
    
    prob = 1
    
    if not isinstance(eta, (list, np.ndarray)):
        for i in range(len(R)):
            if LG_response[i]: 
                prob *= eta
            if CE_response[i]:
                prob *= beta
            if correct_response[i]:
                prob *= 1 - beta
            if incorrect_response[i]:
                prob *= 1 - eta
    else: 
        for i in range(len(R)):
            if LG_response[i]: 
                prob *= eta[i]
            if CE_response[i]:
                prob *= beta[i]
            if correct_response[i]:
                prob *= 1 - beta[i]
            if incorrect_response[i]:
                prob *= 1 - eta[i]
    
    return prob

def compute_rho(R, K_matrix, beta, eta):
    # Computes the probability of a response pattern R given a set of knowledge states K_matrix
    if len(K_matrix.shape) == 1: # reshape if K is a flattened array
        K_matrix = K_matrix.reshape(-1, len(R))
    
    p_K = 1 / K_matrix.shape[0]  # Uniform distribution for prior state probabilities p(K)

    # Sum over all knowledge states
    rho_R = 0
    for K in K_matrix:
        r_value = compute_r(R, K, beta, eta)
        rho_R += r_value * p_K
    
    return rho_R

def generate_powerset(n):
    # Generate the list of all possible combinations of 0s and 1s of length n
    powerset = [np.array([int(x) for x in format(i, '0' + str(n) + 'b')]) for i in range(2 ** n)]
    return powerset

def sample_with_blim(K, num_samples=10, p_k=None, beta=None, eta=None, seed=None, factor=None):
    """
    Sample response patterns from a knowledge structure using the Basic Local Independence Model (BLIM).

    :param K: Knowledge structure represented as a binary matrix (n x m)
    :param num_samples: Number of response patterns to sample
    :param p_k: Probabilities of knowledge states (default is uniform)
    :param beta: Probabilities of a careless error (default is 0.1 for all items)
    :param eta: Probabilities of a lucky guess (default is 0.1 for all items)
    :param seed: Seed for random number generation (optional)
    :return: List of sampled response patterns represented as binary vectors
    """
    n = K.shape[1]
    patterns = []
    powerset_arr_lst = generate_powerset(n)
    
    keys = [''.join(str(x) for x in pattern) for pattern in powerset_arr_lst]
    vals = np.zeros(2**n)
    rho = dict(zip(keys, vals))

    if p_k is None:
        p_k = np.ones(2**n) / 2**n  # Uniform probabilities if not provided

    if beta is None:
        beta = 0.1  # Default beta values
    if factor is not None:
        beta = abs(np.random.normal(beta, beta * factor, n))

    if eta is None:
        eta = 0.1	# Default eta values
    if factor is not None:
        eta = abs(np.random.normal(eta, eta * factor, n))

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    for i, pattern in enumerate(rho.keys()):
        rho[pattern] = compute_rho(powerset_arr_lst[i], K, beta, eta) # Todo: This function assumes Uniform distribution for p(K)! pass distribution instead...
    
    # scale to probabilities
    rho_normalized = {k: v / np.sum(list(rho.values())) for k, v in rho.items()}
    
    for _ in range(num_samples):
        # choose new patterns from powerset according to rho distribution 
        pattern_idx = np.random.choice(2**n, p=list(rho.values()))
        pattern = powerset_arr_lst[pattern_idx]
        patterns.append(pattern)
    
    return patterns

### boolean analysis

def calculate_counterexamples(response_patterns):
    random.seed(42)
    """
    Calculate the number of counterexamples in a matrix similar to the IITA approach.

    :param response_patterns: List of response patterns represented as binary vectors
    :return: Matrix of counterexamples (n x m)
    """
    m = len(response_patterns[0])  # Number of items

    counterexamples = np.zeros((m, m), dtype=float)

    for i in range(m):
        for j in range(i + 1, m):
            for pattern in response_patterns:
                if pattern[i] == 0 and pattern[j] == 1:
                    counterexamples[i][j] += 1
                elif pattern[i] == 1 and pattern[j] == 0:
                    counterexamples[j][i] += 1

    return counterexamples

def generate_all_surmise_relations(n, antisymmetric=False):
    """Generate all valid surmise relations for a given size n."""
    if n < 2 or n > 5:
        raise ValueError("n must be between 2 and 5")

    valid_matrices = []
    total_matrices = 2 ** (n * n)

    for matrix_tuple in tqdm(product([0, 1], repeat=n*n), total=total_matrices, desc="Processing"):
        matrix = np.array(matrix_tuple).reshape(n, n)
        # Check reflexivity, transitivity, antisymmetry
        if np.all(np.diag(matrix) == 1) and is_transitive(matrix):
            if antisymmetric and not is_antisymmetric(matrix):
                continue
            valid_matrices.append(matrix)

    return valid_matrices

### Source: Ünlü, A., & Schrepp, M. (2017). Techniques for sampling quasi-orders. Arch. Data Sci. A, 2, 163-182. 
def extend_quasi_order(base_quasi_order):
    """Extend a given quasi-order by one item, ensuring transitivity."""
    n = base_quasi_order.shape[0] + 1
    extended_matrices = []
    # Generate all possible extensions by adding one row and one column
    for extension in product([0, 1], repeat=2*(n-1)):
        extended_matrix = np.zeros((n, n), dtype=int)
        extended_matrix[:n-1, :n-1] = base_quasi_order
        # Set reflexivity for the new item
        extended_matrix[-1, -1] = 1
        # Fill in the new row and column
        extended_matrix[-1, :n-1] = extension[:n-1]
        extended_matrix[:n-1, -1] = extension[n-1:]
        if is_transitive(extended_matrix):
            extended_matrices.append(extended_matrix)
    return extended_matrices

def sample_surmise_relations(base_set, num_samples=10, antisymmetric=False, seed=None):
    """Sample surmise relations for a given domain size n and a corresponding base set for n-1 items using the Inductive Uniform Extension Approach."""
    if seed is not None:
        np.random.seed(seed)

    sampled_relations = []
    l = len(base_set)
    for base_matrix in tqdm(base_set, desc=f"Sampling from base set (len {l})"):
        # Extend each base quasi-order
        extended_matrices = extend_quasi_order(base_matrix)
        if antisymmetric:
            # Filter for antisymmetry if required
            extended_matrices = [m for m in extended_matrices if is_antisymmetric(m)]

        sampled_relations.extend(extended_matrices)
    
    # Sample the specified number of extended quasi-orders without replacement
    if len(sampled_relations) > num_samples:
        indices = np.random.choice(len(sampled_relations), size=num_samples, replace=False)
        sampled_relations = [sampled_relations[i] for i in indices]
    
    return sampled_relations


# test 
def test_sampling(seed=42, n_samples=num_prosets[4], max_items=7, min_items=2, antisymmetric=False, plot=True): 
    base_set = [
        np.array([[1, 0], [0, 1]]),  # Identity matrix (each item is independent)
        np.array([[1, 1], [0, 1]]),  # Item 1 implies Item 2
        np.array([[1, 0], [1, 1]]),  # Item 2 implies Item 1
        np.array([[1, 1], [1, 1]]),  # Both items are equivalent
    ]
    dataset = []
    if min_items == 2:
        dataset.append(base_set)
    
    for i in range(3, max_items + 1):
        start_time = time.time()
        base_set = sample_surmise_relations(base_set=base_set, num_samples=n_samples, antisymmetric=antisymmetric, seed=seed)
        elapsed_time = time.time() - start_time
        if i >= min_items:
            dataset.append(base_set)
        print(f"Sampling on {i} items elapsed time: {elapsed_time:.2f} seconds --- {len(base_set)} structures generated.")
        
    if plot:
        print()
        # plot histogram for each num of items
        for i, base_set in enumerate(dataset):
            if len(base_set) > 0:
                n = i + min_items
                # plot histogram
                plt.hist([np.sum(m) for m in base_set], bins=50)
                plt.title(f'Number of items: {n}, Number of structures: {len(base_set)}/{num_prosets[n]}') 
                plt.xlabel("Cardinality of Surmise Relations")
                plt.ylabel("Frequency")
                plt.show()
        
        # plot histogram of complete dataset
        all_base_set = [m for base_set in dataset for m in base_set]
        plt.hist([np.sum(m) for m in all_base_set], bins=50)
        plt.title(f'All structures, Number of structures: {len(all_base_set)}/{sum(list(num_prosets.values())[min_items:len(dataset)+min_items])}')
        plt.xlabel("Cardinality of Surmise Relations")
        plt.ylabel("Frequency")
        plt.show()

    # return dataset
