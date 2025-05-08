import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from gen_mol_graph import sdf2graph, mol_to_single_emb
from configs import Model_config
from dataset import Dataset
from torch.autograd import Variable
from collator import collator

# Load model configuration
config = Model_config()

# Check if MPS (Apple GPU) is available
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS (Metal Performance Shaders) backend is not available. Please ensure you are running on an Apple Silicon device with PyTorch configured for MPS.")

device = torch.device("mps")

def drug_embedding(id):
    # Create drug embedding from its SDF file (graph, node, edge attributes, etc.)
    x, edge_attr, edge_index = sdf2graph(id)
    N = x.size(0)
    x = mol_to_single_emb(x)

    # Node adjacency matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # Edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = mol_to_single_emb(edge_attr) + 1

    # Shortest path and other required calculations
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    node = x
    in_degree = adj.long().sum(dim=1).view(-1)
    out_degree = adj.long().sum(dim=0).view(-1)
    edge_input = torch.from_numpy(edge_input).long()

    return node, attn_bias, spatial_pos, in_degree, out_degree, edge_input

# Define a simple prediction function that takes two drug IDs
def predict_interaction(drug1_id, drug2_id, model):
    # Get embeddings for both drugs
    d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input = drug_embedding(drug1_id)
    p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input = drug_embedding(drug2_id)

    # Predicting interaction using the model
    model.eval()
    with torch.no_grad():
        # Send data to device (Apple MPS or CPU)
        d_node = d_node.to(device)
        d_attn_bias = d_attn_bias.to(device)
        d_spatial_pos = d_spatial_pos.to(device)
        d_in_degree = d_in_degree.to(device)
        d_out_degree = d_out_degree.to(device)
        d_edge_input = d_edge_input.to(device)

        p_node = p_node.to(device)
        p_attn_bias = p_attn_bias.to(device)
        p_spatial_pos = p_spatial_pos.to(device)
        p_in_degree = p_in_degree.to(device)
        p_out_degree = p_out_degree.to(device)
        p_edge_input = p_edge_input.to(device)

        # Forward pass
        score = model(
            d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input,
            p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input
        )

        # Get the prediction
        prediction = score.argmax(dim=1).cpu().numpy()[0] + 1  # Assuming the output is a class label

    return prediction

def main():
    # Input: Drug IDs (drug1_id, drug2_id)
    drug1_id = input("Enter the first drug ID: ")
    drug2_id = input("Enter the second drug ID: ")

    # Load pre-trained model
    model = torch.load('save_model/best_model.pth', map_location=device)
    model = model.to(device)

    # Make a prediction for the interaction between the two drugs
    interaction = predict_interaction(drug1_id, drug2_id, model)

    # Output the prediction result
    if interaction == 1:  # Assuming '1' means interaction occurs
        print(f"Interaction detected between Drug {drug1_id} and Drug {drug2_id}.")
    else:
        print(f"No interaction detected between Drug {drug1_id} and Drug {drug2_id}.")

if __name__ == "__main__":
    main()
