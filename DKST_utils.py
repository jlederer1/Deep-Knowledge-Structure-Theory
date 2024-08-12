# imports 
import os
import json
import pickle
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from tqdm import tqdm

from KST_utils import sample_surmise_relations, surmise_matrix_to_states, calculate_counterexamples, sample_with_blim, generate_powerset, R_to_matrix
from DKST_models import generate_sequence
from learning_spaces.kst import iita

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



# sequences 
def set_of_states_in(sequences):
    return {sublst for lst in sequences for sublst in lst}

def sort_sequences_alphabetic(set_of_states):
    # sorts list of strings a) by length b) alphabetically like ""<"a"<"b"<"c"<"ab"<"ac"<"bc"<"abc"<"<pad>"
    sorted_states = sorted(set_of_states, key=lambda x: (len(x), x))
    return sorted_states

def create_state2idx(n, eos_token="<eos>", pad_token="<pad>"):   
    # set_of_states = set_of_states_in(sequences)
    # if pad_token in set_of_states:
    #     set_of_states.remove(pad_token)
        
    
    domain = [chr(97 + i) for i in range(n)]  # ASCII value of 'a' is 97
    powerset = set()
    for i in range(1 << n):  # Equivalent to 2^n
        subset = ""
        for j in range(n):
            # Check if the j-th element is in the i-th subset
            if i & (1 << j):
                subset += domain[j]
        powerset.add(subset)

    sorted_set_of_states = sort_sequences_alphabetic(list(powerset))
    state2idx = {state: idx for idx, state in enumerate(sorted_set_of_states)}
    
    state2idx[eos_token] = 2**n 
    state2idx[pad_token] = 2**n + 1
    
    return state2idx

def encode_sequence(sequence, state2idx):
    return [state2idx[state] for state in sequence]

def encode_sequences(sequences, state2idx):
    return np.array([
        encode_sequence(sequence, state2idx)
        for sequence in tqdm(sequences)
    ])

# knowledge structures 
def binary_sort(array):
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected a numpy array, but got {type(array)} instead")
        return []
    
    # Function to convert binary list to integer
    def binary_to_int(binary_list):
        return int("".join(str(x) for x in binary_list), 2)

    # Sort the array based on two criteria:
    # 1. The sum of the elements in the sublist.
    # 2. The integer value represented by the binary number in the sublist.
    return np.array(sorted(array, key=lambda sublist: (sum(sublist), -binary_to_int(sublist))))

def sort_structures(structures):
    return [binary_sort(structure) for structure in structures]

def binary_to_alphabetic(seq):
    if isinstance(seq[0], np.ndarray):
        seq = ["".join(str(i) for i in state) for state in seq]
    # transforms states from binary to alphabetic set representation:
    # ["00", "10", "11"] -> ["", "a", "ab"]
    # ["000", "001", "100", "101", "111"] -> ["", "c", "ac", "abc"]
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    alphabet = [c for c in alphabet]
    result = []
    for state in seq:
        new_state = ""
        for idx, item in enumerate(state):
            if item == "1":
                new_state += alphabet[idx]
        result.append(new_state)
    return result

def is_closed_under_union(states):
    for s1 in states:
        for s2 in states:
            if set(s1).union(set(s2)) not in [set(s) for s in states]:
                return False
    return True

def is_closed_under_intersection(states):
    for s1 in states:
        for s2 in states:
            if set(s1).intersection(set(s2)) not in [set(s) for s in states]:
                return False
    return True

# counter-examples
def normalize_Bij(Bij):
    # log where Bij > zero
    mask = Bij > 0
    Bij[mask] = np.log(Bij[mask])
    Bij = Bij.astype(float)

    # normalize
    # Bij /= np.sum(Bij)
    # Bij *= len(Bij) ** 2

    # normalize the range of the non-diagonal values to [0, 1]
    min_value = np.min(Bij[mask])
    max_value = np.max(Bij[mask])
    Bij[mask] = (Bij[mask] - min_value) / (max_value - min_value)

    return Bij

# padding 
def pad_relations(matrices):
    # pad 2D matrixes to N x N, such that [[1,2],[3,4]] -> [[1,2,0,0], [3,4,0,0], [0,0,0,0], [0,0,0,0]] for N=4
    N = max(len(matrix) for matrix in matrices)

    padded_matrices = []

    for matrix in matrices:
        padded_matrix = []
        for i in range(N):
            if i < len(matrix):
                row = matrix[i] + [0]*(N-len(matrix[i]))
            else:
                row = [0]*N
            padded_matrix.append(row)
        padded_matrices.append(padded_matrix)

    return padded_matrices

def pad_sequences(sequences, pad_token, vocab_size):
    # max_length = max(len(seq) for seq in sequences)   
    padded_sequences = [
        seq + [pad_token] * (vocab_size - len(seq))
        for seq in sequences
    ]
    return padded_sequences

def pad_adj_matrix(matrix, n):
    # pad 2D matrix to N x N, such that [[1,2],[3,4]] -> [[1,2,0,0], [3,4,0,0], [0,0,0,0], [0,0,0,0]] for N=4
    padded_matrix = np.zeros((n,n), dtype=matrix.dtype)
    padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
        
    return padded_matrix

def pad_counter_examples(counter_examples, n): 
    # matrices: list of lists, each containing a number of unpadded arrays
    padded_matrices = []
    for CE_lst in counter_examples:
        padded_matrices.append([pad_adj_matrix(matrix, n) for matrix in CE_lst])
    return padded_matrices

# data aggregation by frequency of response patterns
def calculate_counts(response_patterns, state2idx, standadize=False):
    """
    Calculate the counts of all response patterns in a matrix for aggregating the dataset of response patterns.

    :param response_patterns: List of response patterns represented as binary vectors
    :param state2idx: Dictionary mapping states to indices, yields order for the data aggregation
    :return: Array of counts (2 ** num_items) associated to all possible response patterns ordered lexicographically
    """
    
    m = len(response_patterns[0])  # Number of items
    counts = np.zeros(2 ** m, dtype=int)  # Initialize counts array
    
    response_patterns_alphabetic = binary_to_alphabetic(["".join(str(i) for i in R) for R in response_patterns])

    for pattern in response_patterns_alphabetic:
        idx = state2idx[pattern]
        # Increment the count for this index
        counts[idx] += 1

    if standadize:
        counts = (counts - counts.mean()) / counts.std()
    
    return counts

# plot
def plot_prediction(K, dataset, C_np, prediction, method, threshold=None):
    colors = []
    for t in dataset.state2idx.values():
        if t in K and t in prediction: # TP - correct
            colors.append('green')
        elif t in K and t not in prediction: # FN - missing
            colors.append('salmon')
        elif t not in K and t in prediction: # FP - extra
            colors.append('red')
        else:
            colors.append('lightgreen') # TN - correct
    
    shapes = []
    for i, color in enumerate(colors):
        if color in ['red', 'salmon']:
            shapes.append(
                dict(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=i,
                    y0=0,
                    x1=i,
                    y1=1,
                    line=dict(
                        color="purple",
                        width=1,
                        dash="dash",
                    )
                )
            )
    
    if threshold is not None:
        shapes.append(dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=threshold, y1=threshold, line=dict(color='blue', width=2)))


    fig = go.Figure(data=go.Bar(x=list(dataset.state2idx.keys())[:-2], y=C_np,
                                hovertext=list(dataset.state2idx.keys())[:-2],
                                marker_color=colors))  
    fig.update_layout(title={
                        'text': "Barplot of (Standardized) State Frequencies",
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    xaxis_title='State',
                    yaxis_title='Frequency', 
                    shapes=shapes,
                    autosize=True,
                    margin=dict(
                        l=50,  # left margin
                        r=50,  # right margin
                        b=90,  # bottom margin
                        t=90,  # top margin
                        pad=10  # padding
                    ),
                    annotations=[
                        dict(
                            x=0.5,
                            y=1.1,
                            showarrow=False,
                            text=f"{method} vs. Ground truth",
                            xref="paper",
                            yref="paper",
                            font=dict(size=14, color='rgb(37,37,37)'),
                            bgcolor='rgba(255, 255, 255, 0.95)'
                        ),
                        dict(
                            x=0.5,
                            y=-0.35,
                            showarrow=False,
                            text="This plot shows the (standardized) state frequencies used as input to the regression module. \nBars are colored based on TP (green), TN, (light green), FP (red), and FN (salmon).",
                            xref="paper",
                            yref="paper"
                        )
                    ])
    fig.show()

def plot_feature_vector(feature_vector):
    # Normalize the feature vector to be between -1 and 1
    min_val, max_val = np.min(feature_vector), np.max(feature_vector)
    mean_val = (max_val + min_val) / 2.0
    range_val = max_val - min_val
    normalized_vector = (feature_vector - mean_val) / range_val * 2
    
    # Use a diverging colormap that transitions smoothly
    cmap = plt.get_cmap('coolwarm')
    
    # Reshape the feature vector for visualization
    image = normalized_vector.reshape(1, -1)

    # Plot the feature vector
    plt.figure(figsize=(10, 1))  # Adjust the figure size to make it more like a long horizontal bar
    im = plt.imshow(image, aspect='auto', cmap=cmap, interpolation='nearest')
    plt.colorbar(im, orientation='vertical')  # Show the color scale
    plt.axis('off')  # Turn off axis labels
    plt.title('Feature Vector Visualization (Horizontal)')
    plt.show()

# pytorch dataset 
class DKSTDataset(Dataset):
    def __init__(self, config_path, baseset_path=None): 
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}.")
        else:
            data_dir = os.path.dirname(os.path.dirname(config_path))
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.d_type = getattr(torch, self.config['d_type'])
            
            # load relations & sequences, build and save if not found
            if f"{self.config['SR_ID']}.json" not in os.listdir(f"{data_dir}\\raw") or f"{self.config['KS_ID']}.pkl" not in os.listdir(f"{data_dir}\\raw"):
                relations = self.build_relations(baseset_path=baseset_path, data_dir=data_dir)
                self.knowledge_structures = self.build_sequences(relations, save_path=data_dir)
            df = pd.read_json(f"{data_dir}\\raw\\{self.config['SR_ID']}.json", orient="records")  
            relations = df['Relations']
            sequences = df['Sequences']
            
            powerset = generate_powerset(self.config['max_items'])
            self.powerset = binary_sort(np.vstack(powerset))
            
            self.relations = pad_relations(relations)
            self.state2idx = create_state2idx(self.config["max_items"], self.config['EOS_TOKEN'], self.config['PAD_TOKEN'])
            self.idx2state = {idx: state for state, idx in self.state2idx.items()}
            self.vocab_size = len(self.state2idx.values())
            padded_sequences = pad_sequences(sequences, self.config['PAD_TOKEN'], self.vocab_size)
            self.encoded = encode_sequences(padded_sequences, self.state2idx)
            
            file_name = self.config['KS_ID']  + ".pkl"
            with open(f"{data_dir}/raw/{file_name}", 'rb') as f:
                self.knowledge_structures = pickle.load(f)
            # load responses, build and save if not found
            if f"{self.config['CE_ID']}.pkl" not in os.listdir(f"{data_dir}\\raw"):
                self.build_counterexamples(self.knowledge_structures, save_path=data_dir) 
            
            file_name = "C" + self.config['CE_ID'][2:] + ".pkl"
            with open(f"{data_dir}\\raw\\{file_name}", 'rb') as f:
                counts = pickle.load(f)
            self.counts = counts
            
            file_name = self.config['CE_ID']  + ".pkl"
            with open(f"{data_dir}\\raw\\{file_name}", 'rb') as f:
                counter_examples = pickle.load(f)
            # Convert lists back to arrays
            counter_examples = [[np.array(C_list) for C_list in counter_examples_per_relation] for counter_examples_per_relation in counter_examples]
            self.C = pad_counter_examples(counter_examples, self.config['max_items'])
            # print(f"DEBUG: {len(self.C)}, {len(self.C[0])}, {type(self.C[0][0])}, {self.C[0][0]}")

    def __getitem__(self, i):
        S = torch.tensor(np.array([self.relations[i]]).flatten(), dtype=self.d_type)
        input_seq, target_seq = torch.tensor(self.encoded[i, :-1], dtype=self.d_type).long(), torch.tensor(self.encoded[i, 1:], dtype=self.d_type).long()
        # C_tuple = tuple(torch.tensor(c.flatten(), dtype=self.d_type) for c in self.C[i])
        # C = torch.tensor(random.choice(self.C[i]).flatten(), dtype=self.d_type)
        C = torch.tensor(random.choice(self.counts[i]), dtype=self.d_type)

        return S, input_seq, target_seq, C

    def __len__(self):
        return len(self.relations)
    
    def build_relations(self, baseset_path=None, baseset_n=7, data_dir="data"):
        dataset = []
        if baseset_path == None:
            base_set = [
                np.array([[1, 0], [0, 1]]),  # Identity matrix (each item is independent)
                np.array([[1, 1], [0, 1]]),  # Item 1 implies Item 2
                np.array([[1, 0], [1, 1]]),  # Item 2 implies Item 1
                np.array([[1, 1], [1, 1]]),  # Both items are equivalent
            ]
            if self.config['antisymmetric']:
                base_set = base_set[:-1]
            if self.config["min_items"] == 2:
                dataset.append(base_set)
        
            for i in tqdm(range(3, self.config["max_items"] + 1)):
                base_set = sample_surmise_relations(base_set=base_set, num_samples=self.config["n_samples"], antisymmetric=self.config["antisymmetric"], seed=self.config["seed"])
                if i >= self.config["min_items"]:
                    dataset.append(base_set)
        
        elif baseset_n==7: # path = "SR_max7_min7_n50000_prosets_72.json"
            base_set = pd.read_json(f"{data_dir}\\raw\\{baseset_path}.json", orient="records")['Relations']
            base_set = [np.array(r) for r in base_set]
            next_extend = sample_surmise_relations(base_set=base_set, num_samples=self.config["n_samples"], antisymmetric=self.config["antisymmetric"], seed=self.config["seed"])
            dataset.append(next_extend)
    
        return dataset
        # torch.save(dataset, f"{self.config['S_id']}.pth")

    def build_sequences(self, relations, save_path="data"):
        relations = [r for domain in relations for r in domain]
        structures = [surmise_matrix_to_states(r) for r in relations] 
        sorted_structures = sort_structures(structures)
        # save to pickle
        file_name = self.config['KS_ID'] + ".pkl"
        path = f"{save_path}\\raw"
        if not os.path.exists(path): os.makedirs(path)
        if os.path.exists(f"{path}\\{file_name}"):
            os.remove(f"{path}\\{file_name}")
        with open(f"{path}\\{file_name}", 'wb') as f:
            pickle.dump(sorted_structures, f)
        
        sequences = [[''.join(map(str, state)) for state in structure] for structure in sorted_structures]
        sequences = [binary_to_alphabetic(seq) + [self.config['EOS_TOKEN']] for seq in sequences] # add eos_toeken to each sequence

        df = pd.DataFrame()
        df['Relations'] = relations  
        df['Sequences'] = sequences
        
        # save to json
        file_name = self.config['SR_ID'] + ".json"
        if os.path.exists(f"{path}\\{file_name}"):
            os.remove(f"{path}\\{file_name}")
        df.to_json(f"{path}\\{file_name}", orient="records")
        
        return sorted_structures 
    
    def build_counterexamples(self, knowledge_structures, save_path="data"):   ### Todo:  Change to "build_response_data"
        counter_examples = [[] for _ in range(len(knowledge_structures))] 
        counts_per_relation = [[] for _ in range(len(knowledge_structures))]
        
        for j, size in enumerate(self.config['sample_sizes']):
            for i in tqdm(range(len(knowledge_structures)), desc=f"Generating dataset part ({j+1}/{len(self.config['sample_sizes'])})"):
                # print(f"DEBUG: {type(knowledge_structures[i])} {knowledge_structures[i]}")
                
                beta = abs(np.random.normal(loc=0.12, scale=0.06))  # random.choice(self.config['beta'])
                eta = abs(np.random.normal(loc=0.12, scale=0.06))  # random.choice(self.config['eta'])
                
                samples = sample_with_blim(knowledge_structures[i], size, p_k=None, beta=beta, eta=eta, seed=self.config['seed'])
                # calculate data aggregate by freqeuncies of response patterns
                counts = calculate_counts(samples, self.state2idx, standadize=self.config['standadized'])
                counts_per_relation[i].append(counts)
                
                C = calculate_counterexamples(samples)
                if self.config['normalized']:
                    C = normalize_Bij(C)
                counter_examples[i].append(C)
        
        # save counterexamples to pickle
        path = f"{save_path}\\raw"
        if not os.path.exists(path): os.makedirs(path)
        
        file_name = self.config['CE_ID'] + ".pkl"
        if os.path.exists(f"{path}\\{file_name}"):
            os.remove(f"{path}\\{file_name}")
        with open(f"{path}\\{file_name}", 'wb') as f:
            pickle.dump(counter_examples, f)
        
        # save counts to pickle
        file_name = "C" + self.config['CE_ID'][2:] + ".pkl"
        if os.path.exists(f"{path}\\{file_name}"):
            os.remove(f"{path}\\{file_name}")
        with open(f"{path}\\{file_name}", 'wb') as f:
            pickle.dump(counts_per_relation, f)
        
        return counter_examples, counts_per_relation

    def split(self, dataset, phase=1, seed=None): 
        # Split the dataset into train, val and test sets
        if phase == 1:
            train_size, val_size, test_size = self.config['phase_1_split']
        else: 
            train_size, val_size, test_size = self.config['phase_2_split']
        
        if seed: 
            torch.manual_seed(seed)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        return train_dataset, val_dataset, test_dataset
    
    def test_item(self, i, model, regressor, samples_size=500, beta = 0.1, eta = 0.1, plot=False, print_performance=False, factor=None):
        items = [(s,idx) for s, idx in self.state2idx.items() if s not in [self.config['EOS_TOKEN'], self.config['PAD_TOKEN']]]
        ### Get target Knowledge Structure and generate dataset
        s = np.array([self.relations[i]])
        S = torch.tensor(s.flatten(), dtype=self.d_type)
        k = surmise_matrix_to_states(s[0])
        k_alphabetic = binary_to_alphabetic(["".join(str(item) for item in row) for row in k])

        if print_performance: print(f"Target:              {k_alphabetic}")
        
        try: 
            samples = sample_with_blim(k, samples_size, p_k=None, beta=beta, eta=eta, seed=None, factor=factor)
            D = np.vstack(samples)
            response = iita(D, v=1)
        except ValueError as e:
            print(f"Error: {e}")
            print("retry sampling...")
            samples = sample_with_blim(k, samples_size, p_k=None, beta=beta, eta=eta, seed=None)
            D = np.vstack(samples)
            response = iita(D, v=1)
        k_iita = surmise_matrix_to_states(R_to_matrix(response['implications'] + [(i,i) for i in range(k.shape[1])]))
        k_iita_aphabetic = binary_to_alphabetic(["".join(str(item) for item in row) for row in k_iita])
        sym_diff_iita = len(set(k_alphabetic) ^ set(k_iita_aphabetic))
        length_diff_iita = len(k_iita_aphabetic) - len(k_alphabetic)
        if print_performance: print(f"IITA (diff {sym_diff_iita}):       {k_iita_aphabetic}")
        
        C = calculate_counts(samples, self.state2idx, standadize=self.config['standadized'])
        C_torch = torch.tensor(C, dtype=self.d_type).to(model.config['device'])
        model.eval()
        regressor.eval()
        eos_token_id = self.state2idx[self.config['EOS_TOKEN']]
        pad_token_id = self.state2idx[self.config['PAD_TOKEN']]
        input_matrix = S.unsqueeze(0) # batching
        data_embedding = regressor(C_torch)
        
        generated_sequence, prediction_embedding, _ = generate_sequence(model, input_matrix, self, eos_token_id=eos_token_id, data_embedding=data_embedding, plot=None)
        gen_seq_lst = [t for t in generated_sequence if t != pad_token_id and t != eos_token_id]
        prediction_alphabetic = [self.idx2state[t] for t in gen_seq_lst]
        # Calculate Symmetric Difference between generated and target sequence
        sym_diff_prediction = len(set(k_alphabetic) ^ set(prediction_alphabetic))
        length_diff_DL = len(prediction_alphabetic) - len(k_alphabetic)
        if print_performance: print(f"Prediction (diff {sym_diff_prediction}): {prediction_alphabetic} \n")
        
        # step 2: find all "edge states" with count inbetween max and min values
        # step 3: contstruct all possible structures K resulting from setting a threshold inbetween each pair of unique counts assocciated to the edge states 
        #         (i.e. between every two edge states si, sj | there is no sx with count(si) < count(sx) < count(sj))
        # step 4: calculate sym. diff. between each K and target K, and select the one with the lowest sym. diff.
        
        # Determine optimal "naive" prediction 
        # (i.e. an optimal K w.r.t. avr. sym. diff. -- FN wheighing more than FP,
        #       by setting a threshold on state frequencies/counts)
        # step 1: find min count of all states, max count of all non-states according to target K 
        min_state_count = min([C[self.state2idx[state]] for state in k_alphabetic])
        max_non_state_count = max([C[self.state2idx[state]] for state in self.state2idx if state not in k_alphabetic + [self.config['EOS_TOKEN'], self.config['PAD_TOKEN']]])
        if min_state_count < max_non_state_count:
            # Step 2: Find all "edge states" with count in between max and min values
            edge_states = [state for state, idx in items if min_state_count <= C[idx] <= max_non_state_count]
            # Step 3: Construct all possible structures K resulting from setting a threshold in between each pair of unique counts associated to the edge states
            thresholds = sorted(set(C[self.state2idx[state]] for state in edge_states))
            possible_Ks = [[state for state, idx in items if C[idx] > threshold] for threshold in thresholds]                   
            # add empty and full state if not included
            possible_Ks = [set([""] + K + [self.idx2state[self.vocab_size-3]]) for K in possible_Ks] 
            # Step 4: Calculate sym. diff. between each K and target K, and select the one with the lowest sym. diff.
            sym_diffs = [len(set(k_alphabetic) ^ set(K)) for K in possible_Ks]
            optimal_diff = min(sym_diffs)
            optimal_threshold = thresholds[sym_diffs.index(min(sym_diffs))]
            optimal_K_alphabetic = possible_Ks[sym_diffs.index(min(sym_diffs))]
            optimal_K_tokens = [self.state2idx[state] for state in optimal_K_alphabetic]
            #######print(f"DEBUG: diffs: {sym_diffs} possible_Ks: {len(possible_Ks)} {possible_Ks}")
        elif min_state_count > max_non_state_count:
            optimal_diff = 0
            optimal_threshold = np.ceil(np.mean([min_state_count, max_non_state_count]))
            optimal_K_alphabetic = k_alphabetic
            optimal_K_tokens = [self.state2idx[state] for state in optimal_K_alphabetic]
        else:
            thresholds = [min_state_count, min_state_count +1]
            possible_Ks = [[state for state, idx in items if C[idx] > threshold] for threshold in thresholds]                   
            # add empty and full state if not included
            possible_Ks = [set([""] + K + [self.idx2state[self.vocab_size-3]]) for K in possible_Ks] 
            # Step 4: Calculate sym. diff. between each K and target K, and select the one with the lowest sym. diff.
            sym_diffs = [len(set(k_alphabetic) ^ set(K)) for K in possible_Ks]
            optimal_diff = min(sym_diffs)
            optimal_threshold = thresholds[sym_diffs.index(min(sym_diffs))]
            optimal_K_alphabetic = possible_Ks[sym_diffs.index(min(sym_diffs))]
            optimal_K_tokens = [self.state2idx[state] for state in optimal_K_alphabetic]
            #######print(f"DEBUG: diffs: {sym_diffs} possible_Ks: {len(possible_Ks)} {possible_Ks}")
        
        if plot  == "raw":
            C = calculate_counts(samples, self.state2idx, standadize=False)
            C_torch = torch.tensor(C, dtype=self.d_type).to(model.config['device'])
        
        prediction_tokens = gen_seq_lst
        K_tokens = [self.state2idx[state] for state in k_alphabetic]
        analysis_result_tokens = [self.state2idx[state] for state in k_iita_aphabetic]
        C_np = C_torch.cpu().numpy()
        
        #print("\nGround Truth: ", len(K_tokens), K_tokens, "\nIITA: v1      ", len(analysis_result_tokens), analysis_result_tokens, "\nPrediction:   ", len(prediction_tokens), prediction_tokens)

        prediction_embedding_np = prediction_embedding.cpu().detach().numpy()
        
        if plot: 
            plot_prediction(K=K_tokens, dataset=self, C_np=C_np, prediction=analysis_result_tokens, method="IITA v1")
            plot_feature_vector(prediction_embedding_np)
            plot_prediction(K=K_tokens, dataset=self, C_np=C_np, prediction=prediction_tokens, method="Prediction") 
            plot_prediction(K=K_tokens, dataset=self, C_np=C_np, prediction=optimal_K_tokens, method="Optimal Threshold", threshold=optimal_threshold)
        
        union_closed = is_closed_under_union(prediction_alphabetic)
        intersection_closed = is_closed_under_intersection(prediction_alphabetic)
        quasi_ordinal = union_closed and intersection_closed
        
        return sym_diff_iita, sym_diff_prediction, quasi_ordinal, union_closed, length_diff_iita, length_diff_DL, optimal_diff

def collate_fn(batch):
    # batch is a list of tuples from DKSTDataset.__getitem__
    # separate batch into lists for each component of the data
    S, input_seq, target_seq, C = zip(*batch)
    # collate lists into single tensor and move to correct device
    S = torch.stack(S).to("cuda")
    input_seq = torch.stack(input_seq).to("cuda")
    target_seq = torch.stack(target_seq).to("cuda")
    C = torch.stack(C).to("cuda")
    return S, input_seq, target_seq, C


# test 
def test_dataset(config_path="data/config/test_config.json"):
    start_time = time.time()
    dataset = DKSTDataset(config_path)
    elapsed_time = time.time() - start_time
    print(f"Initializing dataset elapsed time: {elapsed_time:.2f} seconds")
    print(f"Dataset length: {len(dataset)}")
    
    S, input_seq, target_seq, C = dataset[np.random.randint(len(dataset))]
    print(f"Relation {S.shape}, Input-Seq {input_seq.shape}, Target-Seq {target_seq.shape}, Data Image {C.shape} \n")
    
    print(S, "\n")
    print(input_seq, "\n")
    print(target_seq, "\n")
    print(C, "\n")

