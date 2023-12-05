import pandas as pd
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

file_path = '/content/drive/MyDrive/Colab Notebooks/creditcard_2023.csv'
data = pd.read_csv(file_path)
#print(data.head())
X = data.drop(['id', 'Class'], axis=1)
y = data['Class']


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


k_values = [3, 5, 7, 9]
distance_metric = 'euclidean'
knn_models = []

for k in k_values:
    print(f"Training KNN Model with K={k} and {distance_metric} distance metric...")
    knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
    knn.fit(X_train, y_train)
    knn_models.append(knn)