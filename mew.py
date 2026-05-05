
#GCN implementation for crop recommendation system

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

# Load dataset
crop = pd.read_csv('Crop_recommendation.csv')
X = crop.iloc[:, :-1].values   # 7 features
Y = crop.iloc[:, -1].values    # crop labels

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels (rice→0, mango→1, wheat→2 ...)
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

crop["label"].unique()

# Connect each sample to its 5 nearest neighbors
adj_matrix = kneighbors_graph(X_scaled, n_neighbors=5, mode='connectivity')

# Convert to edge list format (required by PyTorch Geometric)
adj_coo = adj_matrix.tocoo()
edge_index = torch.tensor(
    np.vstack([adj_coo.row, adj_coo.col]),
    dtype=torch.long
)

# Convert features and labels to tensors
x = torch.tensor(X_scaled, dtype=torch.float)
y = torch.tensor(Y_encoded, dtype=torch.long)

# Create PyTorch Geometric Data object
data = Data(x=x, edge_index=edge_index, y=y)
print(data)
# Output: Data(x=[2200, 7], edge_index=[2, 11000], y=[2200])

# Split data using masks (GCN works on the whole graph at once)
num_nodes = data.num_nodes
indices = torch.randperm(num_nodes)

train_size = int(0.85 * num_nodes)
train_idx = indices[:train_size]
test_idx  = indices[train_size:]

# Create boolean masks
data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
data.train_mask[train_idx] = True
data.test_mask[test_idx]   = True

import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class CropGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CropGCN, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)   # 7 → 64
        self.conv2 = GCNConv(hidden_dim, hidden_dim)  # 64 → 64
        self.conv3 = GCNConv(hidden_dim, output_dim)  # 64 → 22 crops

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)   # Layer 1
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)   # Layer 2
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv3(x, edge_index)   # Output layer
        return F.log_softmax(x, dim=1)

# Initialize model
num_classes = len(le.classes_)   # 22 crop types
model = CropGCN(
    input_dim=7,          # N, P, K, Temp, Humidity, pH, Rainfall
    hidden_dim=64,
    output_dim=num_classes
)

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
best_acc = 0
best_loss=0
# Training loop
for epoch in range(1, 601):
    loss = train()
    if epoch % 20 == 0:
        acc = test()
       
        if acc > best_acc:
            best_acc = max(best_acc, acc)
            best_loss = loss
        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}')

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    model.eval()

    # Normalize input using the same scaler
    sample = scaler.transform([[N, P, K, temperature, humidity, ph, rainfall]])
    sample_tensor = torch.tensor(sample, dtype=torch.float)

    # Add the new sample to the graph as a new node
    new_x = torch.cat([data.x, sample_tensor], dim=0)
    new_node_idx = new_x.shape[0] - 1  # index of new node

    # Connect new node to its 5 nearest neighbors in training data
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(sample, data.x.numpy())[0]
    top5 = np.argsort(similarities)[-5:]  # 5 most similar nodes

    # Build new edges: connect new node to top 5 neighbors
    new_edges_src = torch.tensor(top5, dtype=torch.long)
    new_edges_dst = torch.full((5,), new_node_idx, dtype=torch.long)

    # Add edges both ways (undirected graph)
    extra_edges = torch.stack([
        torch.cat([new_edges_src, new_edges_dst]),
        torch.cat([new_edges_dst, new_edges_src])
    ], dim=0)

    new_edge_index = torch.cat([data.edge_index, extra_edges], dim=1)

    # Run full forward pass on updated graph
    with torch.no_grad():
        temp_data = Data(x=new_x, edge_index=new_edge_index)
        out = model(temp_data)                    # full forward pass
        pred_class = out[new_node_idx].argmax().item()  # get new node prediction

    # Make sure prediction is within valid range
    num_classes = len(le.classes_)
    if pred_class >= num_classes:
        pred_class = pred_class % num_classes     # safety clamp

    return le.inverse_transform([pred_class])[0]

# Example prediction
crop_name = predict_crop(
    N=22, P=28, K=26,
    temperature=27.6,
    humidity=45,
    ph=4.9,
    rainfall=92
)
print(f'Recommended Crop: {crop_name}')


print("\n\n\n\n--- Final Results ---")
print(f'Best Accuracy: {best_acc:.4f} | Loss at Best Accuracy: {best_loss:.4f}')
#gcn implementation for crop recommendation system

#GCN RESULT BEGIN
 PS C:\Users\user\Desktop\my_works\fcn> uv run gcn.py
Data(x=[2200, 7], edge_index=[2, 11000], y=[2200])
Epoch 020 | Loss: 0.4637 | Accuracy: 0.9242
Epoch 040 | Loss: 0.2134 | Accuracy: 0.9727
Epoch 060 | Loss: 0.1467 | Accuracy: 0.9788
Epoch 080 | Loss: 0.1257 | Accuracy: 0.9727
Epoch 100 | Loss: 0.0995 | Accuracy: 0.9788
Epoch 120 | Loss: 0.0877 | Accuracy: 0.9848
Epoch 140 | Loss: 0.0890 | Accuracy: 0.9879
Epoch 160 | Loss: 0.0764 | Accuracy: 0.9818
Epoch 180 | Loss: 0.0684 | Accuracy: 0.9848
Epoch 200 | Loss: 0.0686 | Accuracy: 0.9818
Epoch 220 | Loss: 0.0675 | Accuracy: 0.9879
Epoch 240 | Loss: 0.0568 | Accuracy: 0.9848
Epoch 260 | Loss: 0.0583 | Accuracy: 0.9848
Epoch 280 | Loss: 0.0570 | Accuracy: 0.9879
Epoch 300 | Loss: 0.0617 | Accuracy: 0.9879
Epoch 320 | Loss: 0.0578 | Accuracy: 0.9848
Epoch 340 | Loss: 0.0550 | Accuracy: 0.9848
Epoch 360 | Loss: 0.0480 | Accuracy: 0.9848
Epoch 380 | Loss: 0.0492 | Accuracy: 0.9848
Epoch 400 | Loss: 0.0425 | Accuracy: 0.9909
Epoch 420 | Loss: 0.0439 | Accuracy: 0.9909
Epoch 440 | Loss: 0.0394 | Accuracy: 0.9909
Epoch 460 | Loss: 0.0402 | Accuracy: 0.9879
Epoch 480 | Loss: 0.0449 | Accuracy: 0.9970
Epoch 500 | Loss: 0.0479 | Accuracy: 0.9909
Epoch 520 | Loss: 0.0467 | Accuracy: 0.9909
Epoch 540 | Loss: 0.0379 | Accuracy: 0.9909
Epoch 560 | Loss: 0.0331 | Accuracy: 0.9879
Epoch 580 | Loss: 0.0456 | Accuracy: 0.9879
Epoch 600 | Loss: 0.0413 | Accuracy: 0.9879
Recommended Crop: mango



#GCN RESULT END

#EMSEMBLE MODEL BEGIN
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection

import numpy as np

crop = pd.read_csv('crop_recommendation.csv')
X = crop.iloc[:,:-1].values
Y = crop.iloc[:,-1].values

X

Y

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15)

models = []
models.append(('SVC', SVC(gamma ='auto', probability = True)))
models.append(('svm1', SVC(probability=True, kernel='poly', degree=1)))
models.append(('svm2', SVC(probability=True, kernel='poly', degree=2)))
models.append(('svm3', SVC(probability=True, kernel='poly', degree=3)))
models.append(('svm4', SVC(probability=True, kernel='poly', degree=4)))
models.append(('svm5', SVC(probability=True, kernel='poly', degree=5)))
models.append(('rf',RandomForestClassifier(n_estimators = 21)))
models.append(('gnb',GaussianNB()))
models.append(('knn1', KNeighborsClassifier(n_neighbors=1)))
models.append(('knn3', KNeighborsClassifier(n_neighbors=3)))
models.append(('knn5', KNeighborsClassifier(n_neighbors=5)))
models.append(('knn7', KNeighborsClassifier(n_neighbors=7)))
models.append(('knn9', KNeighborsClassifier(n_neighbors=9)))

vot_soft = VotingClassifier(estimators=models, voting='soft')
vot_soft.fit(X_train, y_train)
y_pred = vot_soft.predict(X_test)

scores = model_selection.cross_val_score(vot_soft, X_test, y_test,cv=5,scoring='accuracy')
print("Accuracy: ",scores.mean())

import pickle
pkl_filename = 'Crop_Recommendation.pkl'
Model_pkl = open(pkl_filename, 'wb')
pickle.dump(vot_soft, Model_pkl)
Model_pkl.close()

import pickle

crop_recommendation_model_path = 'Crop_Recommendation.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

data = np.array([[22,28,26,27.6,45,4.9,92]])
my_prediction = crop_recommendation_model.predict(data)

my_prediction[0]
#EMSEMBLE MODEL END
#ENSEMBLE RESULT
(fcn) PS C:\Users\user\Desktop\my_works\fcn> uv run crop_model.py
Accuracy:  0.9787878787878789




Best Accuracy : 0.9939 | Loss at Best Accuracy: 0.0588

GCN RESULT:
Accuracy      : 99.09%
Precision     : 99.19%
Recall        : 99.09%
F1-Score      : 99.09%
Specificity   : 99.96%
ENSEMBLE RESULT:    
  Accuracy     : 98.48%
  Precision    : 98.56%
  Recall       : 98.48%
  F1-Score     : 98.48%
  Specificity  : 99.93%