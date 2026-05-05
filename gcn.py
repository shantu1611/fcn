import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report
)
from torch_geometric.data import Data

# Load dataset
crop = pd.read_csv('Crop_recommendation.csv')
X = crop.iloc[:, :-1].values
Y = crop.iloc[:, -1].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

crop["label"].unique()

# Build graph
adj_matrix = kneighbors_graph(X_scaled, n_neighbors=5, mode='connectivity')
adj_coo = adj_matrix.tocoo()
edge_index = torch.tensor(
    np.vstack([adj_coo.row, adj_coo.col]),
    dtype=torch.long
)

x = torch.tensor(X_scaled, dtype=torch.float)
y = torch.tensor(Y_encoded, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)
print(data)

# Train/Test split
num_nodes = data.num_nodes
indices = torch.randperm(num_nodes)
train_size = int(0.85 * num_nodes)
train_idx = indices[:train_size]
test_idx  = indices[train_size:]

data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
data.train_mask[train_idx] = True
data.test_mask[test_idx]   = True

import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class CropGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CropGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

num_classes = len(le.classes_)
model = CropGCN(input_dim=7, hidden_dim=64, output_dim=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    out  = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out  = model(data)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = correct / data.test_mask.sum()
    return acc.item()

best_acc  = 0
best_loss = 0

# Training loop
for epoch in range(1, 500):
    loss = train()
    if epoch % 20 == 0:
        acc = test()
        if acc > best_acc:
            best_acc  = max(best_acc, acc)
            best_loss = loss
        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}')

# ============================================================
#                   METRICS CALCULATION
# ============================================================
def calculate_metrics():
    model.eval()
    with torch.no_grad():
        out  = model(data)
        pred = out.argmax(dim=1)

    # Get true and predicted labels for test set
    y_true = data.y[data.test_mask].numpy()
    y_pred = pred[data.test_mask].numpy()

    # 1. Accuracy
    accuracy  = accuracy_score(y_true, y_pred)

    # 2. Precision
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)

    # 3. Recall / Sensitivity
    recall    = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    # 4. F1-Score
    f1        = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 5. Specificity (manual calculation from confusion matrix)
    cm = confusion_matrix(y_true, y_pred)
    specificity_per_class = []
    for i in range(len(cm)):
        TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        FP = cm[:, i].sum() - cm[i, i]
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificity_per_class.append(spec)
    specificity = np.mean(specificity_per_class)

    # Print all metrics
    print("\n" + "=" * 45)
    print("         MODEL EVALUATION METRICS")
    print("=" * 45)
    print(f"  Accuracy     : {accuracy    * 100:.2f}%")
    print(f"  Precision    : {precision   * 100:.2f}%")
    print(f"  Recall       : {recall      * 100:.2f}%")
    print(f"  F1-Score     : {f1          * 100:.2f}%")
    print(f"  Specificity  : {specificity * 100:.2f}%")
    print("=" * 45)

    # Detailed per-class breakdown
    print("\n--- Per Class Report ---")
    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))

    return accuracy, precision, recall, f1, specificity

# Run metrics
accuracy, precision, recall, f1, specificity = calculate_metrics()

# ============================================================

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    model.eval()
    sample = scaler.transform([[N, P, K, temperature, humidity, ph, rainfall]])
    sample_tensor = torch.tensor(sample, dtype=torch.float)

    new_x = torch.cat([data.x, sample_tensor], dim=0)
    new_node_idx = new_x.shape[0] - 1

    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(sample, data.x.numpy())[0]
    top5 = np.argsort(similarities)[-5:]

    new_edges_src = torch.tensor(top5, dtype=torch.long)
    new_edges_dst = torch.full((5,), new_node_idx, dtype=torch.long)

    extra_edges = torch.stack([
        torch.cat([new_edges_src, new_edges_dst]),
        torch.cat([new_edges_dst, new_edges_src])
    ], dim=0)

    new_edge_index = torch.cat([data.edge_index, extra_edges], dim=1)

    with torch.no_grad():
        temp_data = Data(x=new_x, edge_index=new_edge_index)
        out = model(temp_data)
        pred_class = out[new_node_idx].argmax().item()

    num_classes = len(le.classes_)
    if pred_class >= num_classes:
        pred_class = pred_class % num_classes

    return le.inverse_transform([pred_class])[0]

# Example prediction
crop_name = predict_crop(
    N=22, P=28, K=26,
    temperature=27.6,
    humidity=45,
    ph=4.9,
    rainfall=92
)

# Save after training loop
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'label_encoder': le,
    'edge_index': data.edge_index,
    'x': data.x,
}, 'crop_gcn_model.pth')

print("Model saved successfully!")
print(f'\nRecommended Crop: {crop_name}')

# Final Results
print("\n--- Final Results ---")
print(f"Best Accuracy : {best_acc :.4f} | Loss at Best Accuracy: {best_loss:.4f}")
print(f"Accuracy      : {accuracy    * 100:.2f}%")
print(f"Precision     : {precision   * 100:.2f}%")
print(f"Recall        : {recall      * 100:.2f}%")
print(f"F1-Score      : {f1          * 100:.2f}%")
print(f"Specificity   : {specificity * 100:.2f}%")