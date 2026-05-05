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

# ============================================================
#                   METRICS
# ============================================================
from sklearn.metrics import (
    precision_score, recall_score,
    f1_score, confusion_matrix,
    classification_report
)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)

cm = confusion_matrix(y_test, y_pred)
specificity_per_class = []
for i in range(len(cm)):
    TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    FP = cm[:, i].sum() - cm[i, i]
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0
    specificity_per_class.append(spec)
specificity = np.mean(specificity_per_class)

print("=" * 45)
print("      MODEL EVALUATION METRICS")
print("=" * 45)
print(f"  Accuracy     : {accuracy    * 100:.2f}%")
print(f"  Precision    : {precision   * 100:.2f}%")
print(f"  Recall       : {recall      * 100:.2f}%")
print(f"  F1-Score     : {f1          * 100:.2f}%")
print(f"  Specificity  : {specificity * 100:.2f}%")
print("=" * 45)
print(classification_report(y_test, y_pred, zero_division=0))
# ============================================================

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