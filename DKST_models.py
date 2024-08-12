# imports
import os
import json
import math
import numpy as np
import random
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

import pandas as pd
from learning_spaces.kst import iita



# Transformer 

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=5000, dropout_rate=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # calculate positional encodings once in log space
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).long() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # add positional encodings to input
        x = x + self.pe[:,:x.size(1)]
        x = self.dropout(x)
        return x

class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, bias=True):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, bias)
        
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.activation_function = activation
        self.batch_first = batch_first
        self.norm_first = norm_first
        
        self.dropout = nn.Dropout(self.dropout_value)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias)

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True):
        # Adjust for batch_first option if necessary
        if self.batch_first:
            tgt = tgt.transpose(0, 1)
            memory = memory.transpose(0, 1)

        # Self attention
        tgt2, self_attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                                 key_padding_mask=tgt_key_padding_mask,
                                                 need_weights=need_weights)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt) if self.norm_first else self.norm1(tgt)

        # Multi-head attention                 
        tgt2, multihead_attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                                           key_padding_mask=memory_key_padding_mask,
                                                           need_weights=need_weights)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt) if self.norm_first else self.norm2(tgt)

        # Feedforward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt) if self.norm_first else self.norm3(tgt)

        # Adjust for batch_first option if necessary
        if self.batch_first:
            tgt = tgt.transpose(0, 1)

        return tgt, self_attn_weights, multihead_attn_weights

class CustomTransformerDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__(decoder_layer, num_layers, norm)
        self.layers = nn.ModuleList([CustomTransformerDecoderLayer(
            d_model=decoder_layer.d_model,
            nhead=decoder_layer.nhead,
            dim_feedforward=decoder_layer.dim_feedforward,
            dropout=decoder_layer.dropout_value,
            activation=decoder_layer.activation_function
        ) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm if norm is not None else nn.LayerNorm(decoder_layer.d_model)

    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt

        attentions = []  # Liste, um Aufmerksamkeitsgewichte zu speichern

        for mod in self.layers:
            output, self_attn_weights, _ = mod(output, memory, tgt_mask=tgt_mask,
                                               memory_mask=memory_mask,
                                               tgt_key_padding_mask=tgt_key_padding_mask,
                                               memory_key_padding_mask=memory_key_padding_mask)
            attentions.append(self_attn_weights)

        if self.norm:
            output = self.norm(output)

        return output, attentions

class SimpleDecoderModel(nn.Module):
    def __init__(self, config_path):
        super(SimpleDecoderModel, self).__init__()
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}.")
        data_dir = os.path.dirname(os.path.dirname(config_path))
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.d_type = getattr(torch, self.config['d_type'])
        self.input_dim = self.config['n_items'] ** 2
        self.hidden_dim = self.config['hidden_dim']
        self.vocab_size = 2 ** self.config['n_items'] + 2 # number of states + eos and pad tokens
        self.max_seq_len = self.vocab_size - 1 # num_states + eos token
        self.n_heads = self.config['n_heads'] 
        self.n_layers = self.config['n_layers'] 
        self.dropout_value = self.config['dropout']  
        self.activation = self.config['activation']

        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.max_seq_len, dropout_rate=self.dropout_value)
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=self.number_heads, activation="relu", dropout=self.dropout_value)
        # self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_layers)
        self.decoder_layer = CustomTransformerDecoderLayer(d_model=self.hidden_dim, nhead=self.n_heads, activation=self.activation, dropout=self.dropout_value)
        self.transformer_decoder = CustomTransformerDecoder(self.decoder_layer, num_layers=self.n_layers)
        self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size - 1) # exclude padding token from generation

    def forward(self, S, input_seq, data_embedding=None):
        batch_size = S.shape[0]
        seq_length = input_seq.shape[1]

        input_seq = self.embedding(input_seq)
        input_seq = self.pos_encoder(input_seq)
        input_seq = input_seq.permute(1, 0, 2).float()

        # projection into latent space (memory cell)
        if data_embedding is None:
            K_embedding = self.input_proj(S)
        else:
            K_embedding = data_embedding
        memory = K_embedding.view(1, batch_size, self.hidden_dim)
        
        memory = memory.repeat(1, 1, 1).float()

        # masking decoder's self-attention mechanism
        tgt_mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1).to(input_seq.device)
        
        # forward pass through transformer
        output, attention_wheights = self.transformer_decoder(tgt=input_seq, memory=memory, tgt_mask=tgt_mask, memory_mask=None)
        output = self.output_proj(output)

        return output, K_embedding, attention_wheights

# Multi-Layer Perceptron

class RegressionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_val):
        super(RegressionNetwork, self).__init__()
        # Define fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_val)
        # self.bn1 = nn.BatchNorm1d(hidden_size1)

    def forward(self, x):
        # Apply the first fully connected layers with ReLU activation
        x = self.activation(self.fc1(x)) 
        x = self.dropout(x)
        x = self.activation(self.fc2(x)) 
        x = self.dropout(x)
        # Linear activation for regression
        x = self.fc3(x)
        return x

class EnhancedRegressionNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, dropout_val):
        super(EnhancedRegressionNetwork, self).__init__()
        self.layers = nn.ModuleList()
        # First layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        self.layers.append(nn.Dropout(dropout_val))
        # Additional hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            self.layers.append(nn.Dropout(dropout_val))
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.activation = nn.SiLU()  # Swish activation function

    def forward(self, x):
        # Check if input is 1D and if so, add a batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = self.activation(layer(x))
            else:
                x = layer(x)
        # Remove the batch dimension for single sample input to return a 1D tensor
        if x.size(0) == 1:
            x = x.squeeze(0)
        return x

# Training 

class CustomCELoss(nn.Module):
    def __init__(self, label_smoothing):
        super(CustomCELoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, output, labels):
        # masking out padding tokens
        pad_token_id = output.shape[-1]  # identifying pad tokens
        output_flat = output.permute(1, 2, 0)
        mask = (labels != pad_token_id)

        # Cross-entropy losses for each predicted token
        loss = F.cross_entropy(output_flat, labels, ignore_index=pad_token_id, reduction='none', label_smoothing=self.label_smoothing) 
        masked_loss = loss * mask

        # Averaging loss over non-padding tokens
        averaged_loss = masked_loss.sum() / mask.long().sum()
        return averaged_loss

def train(model, train_loader, optimizer, distance, angle, scheduler, clip_norm, dataset, length_wheight, device, knet=None, prediction_only=False):
    model.train()
    if knet is not None and prediction_only is False: knet.train()

    total_loss_CE = 0.
    total_loss_L = 0.
    total_loss_combiened = 0.
    norm = clip_norm

    for relation, data, targets, C in train_loader:
        optimizer.zero_grad()
        if knet:
            output, K_embedding_batch, _ = knet(relation, data, C)
        else: 
            output, K_embedding_batch, _ = model.forward(relation, data)
        
        # Cross-entropy loss
        loss = distance(output, targets) * (1 - length_wheight) # * len(data) # Scale loss by batch size
        total_loss_CE += loss.item()

        # Length loss: Alignment between hidden representation's norms and sequence lengths # (todo: MSE loss with tolerance window)
        seq_lengths = ((targets != dataset.state2idx[dataset.config['PAD_TOKEN']]) & (targets != dataset.state2idx[dataset.config['EOS_TOKEN']])).sum(dim=1).float()
        embedding_norms = K_embedding_batch.norm(dim=1)
        length_loss = angle(embedding_norms, seq_lengths) * length_wheight
        total_loss_L += length_loss.item() # * len(data) # Scale loss by batch size

        loss_combined = loss + length_loss
        total_loss_combiened += loss_combined
        
        loss_combined.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), norm)
        optimizer.step()

    return total_loss_CE / len(train_loader), total_loss_L / len(train_loader), total_loss_combiened / len(train_loader)

def evaluate(model, data_loader, criterion, device, knet=None):
    model.eval()
    if knet is not None: knet.eval()
    total_loss = 0.

    with torch.no_grad():
        for relation, data, targets, C in data_loader:
            if knet is not None:
                output, _, _ = knet.forward(relation, data, C)
            else: 
                output, _, _ = model(relation, data)
            
            loss = criterion(output, targets)
            total_loss += loss.item()  * len(data)

    return total_loss / len(data_loader)

# Inspection

def pad_weights(array_list):
    size = max(len(arr) for arr in array_list)
    return np.stack([np.pad(arr, (0, size - len(arr)), mode='constant', constant_values=0) for arr in array_list])

def plot_attention_weights(attention_weights, sequence, max_len=None, title="Attention Weights"):
    size = len(attention_weights) 
    fig_size = min(8, int(len(attention_weights)/2))
    # put self-attention weights into one array
    attention_weights = pad_weights(attention_weights)
    
    # plot square array as sns heatmap
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(attention_weights, cmap='viridis', annot=True, fmt=".2f", linecolor='black', linewidth=0.5)
    # Hide annotations for cells with values below 0.1
    for text in plt.gca().texts:
        if float(text.get_text()) < 0.1:
            text.set_text("")
    plt.xticks(np.arange(size), sequence[:size])
    plt.yticks(np.arange(size), sequence[:size], rotation=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def generate_sequence(model, S, dataset, eos_token_id, max_length=None, data_embedding=None, plot=False, embedding_only=False):
    # retrieve max seqeunce length from model config
    if max_length is None: max_length = 2 ** model.config['n_items'] + 1 # maximum sequence length the model is trained on (do not change!)
    
    # Initialize generated sequence with index of start token (empty state) 
    generated_sequence = [0] # zero for empty state as start token 
    attention_weights = []
    K_embedding_batch = torch.Tensor()

    model.eval()
    with torch.no_grad():
        # iteratively generate the output sequence 
        for _ in range(max_length):
            # generated sequence so far to tensor
            current_seq = torch.tensor([generated_sequence], dtype=torch.long).long().to('cuda')
            # Predict the next token
            output, K_embedding_batch, attention  = model(S, current_seq, data_embedding) 
            self_attention = attention[0].cpu()
            attention_weights += [self_attention[0,-1,:]]
            # print(f"\nDEBUG attention: \n{attention_weights[-1][0,-1,:].shape}{attention_weights[-1][0,-1,:]}\n")
            
            # add newly predicted token to sequence 
            next_token_id = output[-1].argmax(dim=-1).item()
            generated_sequence.append(next_token_id)
            
            # Break if EOS token is generated
            if next_token_id == eos_token_id:
                break
            if embedding_only:
                break

    if plot:
            # plot attention weights
            attention_weights = [attention_weights.numpy() for attention_weights in attention_weights]
            padded_attention_weights = pad_weights(attention_weights)
            plot_attention_weights(padded_attention_weights, [dataset.idx2state[idx] for idx in generated_sequence], max_len=len(generated_sequence))

    return generated_sequence, K_embedding_batch, attention_weights

def performance(model, testset, dataset, regressor=None, print_n_errors=0, plot=False):
    model.eval()
    if regressor is not None: regressor.eval()
    device = model.config['device']
    eos_token_id = dataset.state2idx[dataset.config['EOS_TOKEN']]
    pad_token_id = dataset.state2idx[dataset.config['PAD_TOKEN']]
    
    # Accumulators for perfromance meassures 
    correct = 0   # accuracy
    norm_len_diffs = []   # length differences between target sequences and norm of knowledge structure embedding
    avr_sym_diffs = []

    # Select a random subset of 100 samples from the testset to calculate accuracy for 
    n_samples = 100
    if len(testset) < n_samples:
        n_samples = len(testset)
        print(f"Warning: only {n_samples} samples avalable for performance test.")
    if print_n_errors > 100:
        n_samples = min(print_n_errors, len(testset))
    subset = random.sample(list(testset), n_samples) 
    
    # Aggregate pairs of mismatching sequences for display after the test
    prints = [f"    Displaying {print_n_errors} mismatching sequences:"]
    # attention_weights = []

    with torch.no_grad():
        for i, (S, _, tgt_seq, C) in enumerate(tqdm(subset, desc=f"Performance test on {n_samples} random test samples")):
            S, tgt_seq, C = S.to(device), tgt_seq.to(device), C.to(device)

            input_matrix = S.unsqueeze(0)
            # plot attention weights only once per performance test
            # plot = (plot and (i+1 == len(subset) - 1))
            if regressor is not None:
                data_embedding = regressor(C)
                generated_sequence, K_embedding_batch, attention_weights = generate_sequence(model, input_matrix, dataset, eos_token_id=eos_token_id, data_embedding=data_embedding, plot=plot)
                K_embedding_batch = K_embedding_batch.unsqueeze(0) 
                plot = False
            else:
                generated_sequence, K_embedding_batch, attention_weights = generate_sequence(model, input_matrix, dataset, eos_token_id=eos_token_id, plot=plot)
                plot = False

            gen_seq_lst = [t for t in generated_sequence[1:] if t != pad_token_id and t != eos_token_id]
            tgt_seq_lst = [t for t in tgt_seq.tolist() if t != pad_token_id and t != eos_token_id]
            
            sym_diff = len(set(gen_seq_lst) ^ set(tgt_seq_lst))
            avr_sym_diffs += [sym_diff]
            
            # Check if generated sequence matches the target sequence
            if gen_seq_lst == tgt_seq_lst:
                correct += 1
            # else display (up to given n) mismatching sequences
            elif (len(prints)-1)/3 < print_n_errors:
                prints.append(f"    {i}) Symmetric Difference: {sym_diff}")
                target_tokens = [dataset.idx2state[t.item()] for t in tgt_seq]
                target_str = "    [Target]     ||" + "|".join([token.ljust(model.config['n_items']) for token in target_tokens]) + "||"
                prints.append(target_str)
                
                seq_tokens = [dataset.idx2state[idx] for idx in generated_sequence]
                seq_str = "    [Prediction] ||" + "|".join([token.ljust(model.config['n_items']) for token in seq_tokens[1:]]) + "||"
                prints.append(seq_str)

            # calculate difference between target sequence length and norm of embedding vectors
            embedding_norms = K_embedding_batch.norm(dim=1)
            tgt_seq = tgt_seq.unsqueeze(0)
            seq_lengths = ((tgt_seq != pad_token_id) & (tgt_seq != eos_token_id)).sum(dim=1).float()
            diff = embedding_norms - seq_lengths
            norm_len_diffs += [diff.item()]
    
    # Display mismatching sequences
    for seq in prints: print(seq)
    # Calculate and display accuracy and mean length difference
    acc = correct / n_samples
    mean_dist = np.mean([abs(diff) for diff in norm_len_diffs])
    std = np.std(norm_len_diffs)
    print(f"Accuracy: {acc}, Avr Symmetric Difference: {np.mean(avr_sym_diffs)} (std {np.std(avr_sym_diffs)})")
    print(f"Mean abs length difference: {mean_dist} (std {std}) \n")
    return acc, np.mean(avr_sym_diffs), std


# test 
def test_simple_decoder_model(config_path, model=None, dataloader=None):
    if model is None:
        model = SimpleDecoderModel(config_path=config_path)
    
    device = model.config['device']
    model.to(device)
    
    # Generate dummy input data
    batch_size = 2
    input_dim = model.input_dim
    seq_length = model.max_seq_len
    print(f"Vocab size of model {model.vocab_size}")
    
    if dataloader is None:
        input_matrix = torch.randn(batch_size, input_dim, device=device)
        tgt_seq = torch.randint(0, model.vocab_size, (batch_size, seq_length), device=device)
        
        # Perform a forward pass
        output, K_embedding, attention = model(input_matrix, tgt_seq)
    else: 
        S, In, Out, C = next(iter(dataloader))
        output, K_embedding, attention = model(S, Out)
        
    # Check output shapes
    assert output.shape == (seq_length, batch_size, model.vocab_size - 1), f"Output shape mismatch: {output.shape}"
    assert K_embedding.shape == (batch_size, model.hidden_dim), f"K_embedding shape mismatch: {K_embedding.shape}"
    
    print("Test passed. Model's forward pass is functional and output shapes are as expected.")

# (delete) 
def eval_performance(model, testset, dataset, regressor=None, print_n_errors=0, plot=False):
    model.eval()
    if regressor is not None: regressor.eval()
    device = model.config['device']
    eos_token_id = dataset.state2idx[dataset.config['EOS_TOKEN']]
    pad_token_id = dataset.state2idx[dataset.config['PAD_TOKEN']]
    
    # Accumulators for perfromance meassures 
    correct = 0   # accuracy
    norm_len_diffs = []   # length differences between target sequences and norm of knowledge structure embedding
    avr_sym_diffs = []
    avr_sym_diffs_iita = []

    # Select a random subset of 100 samples from the testset to calculate accuracy for 
    n_samples = 100
    if len(testset) < n_samples:
        n_samples = len(testset)
        print(f"Warning: only {n_samples} samples avalable for performance test.")
    if print_n_errors > 100:
        n_samples = min(print_n_errors, len(testset))
    subset = random.sample(list(testset), n_samples) 
    
    # Aggregate pairs of mismatching sequences for display after the test
    prints = [f"    Displaying {print_n_errors} mismatching sequences:"]
    # attention_weights = []

    with torch.no_grad():
        for i, (S, _, tgt_seq, C, CE) in enumerate(tqdm(subset, desc=f"Performance test on {n_samples} random test samples")):
            S, tgt_seq, C = S.to(device), tgt_seq.to(device), C.to(device)

            input_matrix = S.unsqueeze(0)
            # plot attention weights only once per performance test
            plot = (plot and (i+1 == len(subset) - 1))
            if regressor is not None:
                data_embedding = regressor(C)
                generated_sequence, K_embedding_batch, attention_weights = generate_sequence(model, input_matrix, dataset, eos_token_id=eos_token_id, data_embedding=data_embedding, plot=plot)
                K_embedding_batch = K_embedding_batch.unsqueeze(0) 
            else:
                generated_sequence, K_embedding_batch, attention_weights = generate_sequence(model, input_matrix, dataset, eos_token_id=eos_token_id, plot=plot)

            gen_seq_lst = [t for t in generated_sequence[1:] if t != pad_token_id and t != eos_token_id]
            tgt_seq_lst = [t for t in tgt_seq.tolist() if t != pad_token_id and t != eos_token_id]
            
            sym_diff = len(set(gen_seq_lst) ^ set(tgt_seq_lst))
            avr_sym_diffs += [sym_diff]
            
            # Check if generated sequence matches the target sequence
            if gen_seq_lst == tgt_seq_lst:
                correct += 1
            # else display (up to given n) mismatching sequences
            elif (len(prints)-1)/3 < print_n_errors:
                prints.append(f"    {i}) Symmetric Difference: {sym_diff}")
                target_tokens = [dataset.idx2state[t.item()] for t in tgt_seq]
                target_str = "    [Target]     ||" + "|".join([token.ljust(model.config['n_items']) for token in target_tokens]) + "||"
                prints.append(target_str)
                
                seq_tokens = [dataset.idx2state[idx] for idx in generated_sequence]
                seq_str = "    [Prediction] ||" + "|".join([token.ljust(model.config['n_items']) for token in seq_tokens[1:]]) + "||"
                prints.append(seq_str)

            # calculate difference between target sequence length and norm of embedding vectors
            embedding_norms = K_embedding_batch.norm(dim=1)
            tgt_seq = tgt_seq.unsqueeze(0)
            seq_lengths = ((tgt_seq != pad_token_id) & (tgt_seq != eos_token_id)).sum(dim=1).float()
            diff = embedding_norms - seq_lengths
            norm_len_diffs += [diff.item()]
    
    # Display mismatching sequences
    for seq in prints: print(seq)
    # Calculate and display accuracy and mean length difference
    acc = correct / n_samples
    mean_dist = np.mean([abs(diff) for diff in norm_len_diffs])
    std = np.std(norm_len_diffs)
    print(f"Accuracy: {acc}, Avr Symmetric Difference: {np.mean(avr_sym_diffs)} (std {np.std(avr_sym_diffs)})")
    print(f"Mean abs length difference: {mean_dist} (std {std}) \n")
    return acc, mean_dist, std
