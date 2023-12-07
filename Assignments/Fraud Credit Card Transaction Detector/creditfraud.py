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


def create_neural_network(hidden_units, initialization):
    model = Sequential()
    model.add(Dense(units=hidden_units, input_dim=X_train.shape[1], activation='relu', kernel_initializer=initialization))
    model.add(Dense(units=1, activation='sigmoid'))
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


loss_zero_init = []
loss_random_init = []

print("Training Neural Network Model with Zero Initialization...")
nn_model_zero_init = create_neural_network(5, initialization='zeros')
history_zero_init = nn_model_zero_init.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), verbose=1)
loss_zero_init = history_zero_init.history['loss']

print("Training Neural Network Model with Small Random Initialization...")
nn_model_random_init = create_neural_network(5,initialization='random_uniform')
history_random_init = nn_model_random_init.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), verbose=1)
loss_random_init = history_random_init.history['loss']

# Zero Initialization Graph
plt.figure(figsize=(8, 6))
plt.plot(loss_zero_init, label='Zero Initialization')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve for Zero Initialization')
plt.legend()
plt.show()

# Random Initialization Graph
plt.figure(figsize=(8, 6))
plt.plot(loss_random_init, label='Random Initialization')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve for Random Initialization')
plt.legend()
plt.show()

# Comparison Graph
plt.figure(figsize=(8, 6))
plt.plot(loss_zero_init, label='Zero Initialization')
plt.plot(loss_random_init, label='Random Initialization')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curves for Different Weight Initialization Methods')
plt.legend()
plt.show()