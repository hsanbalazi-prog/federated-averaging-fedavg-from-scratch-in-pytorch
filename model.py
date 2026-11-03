"""
Federated Averaging (FedAvg) from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_mlp_classifier
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))   # raw logits

def build_mlp_classifier(input_size, hidden_size, num_classes):
    return MLP(input_size, hidden_size, num_classes)

# Step 2 - build_synthetic_dataset
import torch
def build_synthetic_dataset(num_samples,input_size,num_classes,seed,noise=0.1):
    g=torch.Generator().manual_seed(seed)
    centers=torch.randn(num_classes,input_size,generator=g)
    labels=torch.randint(0,num_classes,(num_samples,),generator=g)
    noise_term=noise*torch.randn(num_samples,input_size,generator=g)
    features=centers[labels]+noise_term
    return features.float(),labels.long()

# Step 3 - train_test_split_dataset
import torch

def train_test_split_dataset(features, labels, test_fraction, seed):
    N = features.shape[0]
    n_test = int(N * test_fraction)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)

    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    test_features = features[test_idx]
    test_labels = labels[test_idx]
    train_features = features[train_idx]
    train_labels = labels[train_idx]

    return train_features, train_labels, test_features, test_labels

# Step 4 - partition_data_iid
import torch

def partition_data_iid(features, labels, num_clients, seed):
    num_rows = features.shape[0]
    num_clients = max(1, num_clients)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_rows, generator=generator)

    index_groups = torch.tensor_split(perm, num_clients)

    return [(features[idx], labels[idx]) for idx in index_groups]

# Step 5 - partition_data_non_iid
import torch

def partition_data_non_iid(features, labels, num_clients, shards_per_client, seed):
    num_shards = num_clients * shards_per_client

    sorted_idx = torch.argsort(labels, stable=True)
    shards = torch.tensor_split(sorted_idx, num_shards)

    generator = torch.Generator()
    generator.manual_seed(seed)
    shard_perm = torch.randperm(num_shards, generator=generator).tolist()

    partitions = []
    for c in range(num_clients):
        my_shards = shard_perm[c * shards_per_client:(c + 1) * shards_per_client]
        idx = torch.cat([shards[s] for s in my_shards])
        partitions.append((features[idx], labels[idx]))

    return partitions

# Step 6 - count_client_samples
def count_client_samples(client_partitions):
    return [len(client_labels) for client_features, client_labels in client_partitions]

# Step 7 - iterate_client_batches
import torch

def iterate_client_batches(features, labels, batch_size, seed):
    n = features.shape[0]

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)

    shuffled_features = features[perm]
    shuffled_labels = labels[perm]

    batches = []
    for start in range(0, n, batch_size):
        end = start + batch_size
        batches.append((shuffled_features[start:end], shuffled_labels[start:end]))

    return batches

# Step 8 - compute_batch_loss
import torch.nn.functional as F

def compute_batch_loss(model, batch_features, batch_labels):
    logits = model(batch_features)
    loss = F.cross_entropy(logits, batch_labels)
    return loss

# Step 9 - local_sgd_step
def local_sgd_step(model, optimizer, batch_features, batch_labels):
    optimizer.zero_grad()
    loss = compute_batch_loss(model, batch_features, batch_labels)
    loss.backward()
    optimizer.step()
    return loss.item()

# Step 10 - train_client_local (not yet solved)
# TODO: implement

# Step 11 - clone_model_state (not yet solved)
# TODO: implement

# Step 12 - load_model_state (not yet solved)
# TODO: implement

# Step 13 - initialize_global_state (not yet solved)
# TODO: implement

# Step 14 - add_state_dicts (not yet solved)
# TODO: implement

# Step 15 - scale_state_dict (not yet solved)
# TODO: implement

# Step 16 - aggregate_weighted_average (not yet solved)
# TODO: implement

# Step 17 - select_round_clients (not yet solved)
# TODO: implement

# Step 18 - run_communication_round (not yet solved)
# TODO: implement

# Step 19 - evaluate_accuracy (not yet solved)
# TODO: implement

# Step 20 - run_fedavg (not yet solved)
# TODO: implement

# Step 21 - train_centralized_baseline (not yet solved)
# TODO: implement

# Step 22 - run_fedavg_iid (not yet solved)
# TODO: implement

# Step 23 - run_fedavg_non_iid (not yet solved)
# TODO: implement

# Step 24 - compute_non_iid_gap (not yet solved)
# TODO: implement

# Step 25 - rounds_to_target_vs_local_epochs (not yet solved)
# TODO: implement

# Step 26 - accuracy_vs_client_fraction (not yet solved)
# TODO: implement

