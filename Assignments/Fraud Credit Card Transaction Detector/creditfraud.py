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