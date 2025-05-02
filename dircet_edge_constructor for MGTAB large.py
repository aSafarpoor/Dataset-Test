import torch
from tqdm import tqdm

# Load and collect directed edges from large_edge_index0.pt
edge_index = torch.load("large_edge_index0.pt").t().tolist()
directed_edges = []
for x, y in tqdm(edge_index, desc="Processing edge_index0"):
    directed_edges.append((x, y))  # store as tuple for deduplication
print(len(directed_edges))

# Load and reverse edges from large_edge_index1.pt
edge_index = torch.load("large_edge_index1.pt").t().tolist()
for x, y in tqdm(edge_index, desc="Processing edge_index1"):
    directed_edges.append((y, x))  # reversed edge, stored as tuple
print(len(directed_edges))

# Remove duplicates
unique_edges = list(set(directed_edges))
print(f"Unique directed edges: {len(unique_edges)}")

# Convert back to tensor and save
directed_edge_tensor = torch.tensor(unique_edges, dtype=torch.long).t()
print(directed_edge_tensor.shape)
torch.save(directed_edge_tensor, "directed_edge_index.pt")
