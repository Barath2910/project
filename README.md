# Automated Hypoxia Prevention and Real-Time Human Life Monitoring System in Car
## Hardware Requirements:
1.PC Processor - 11th Gen Intel
2.System type :64-bit processor
## SoftwarenRequirements:
1. Jupyter Notebook
2. Python Libraries
## Implementation:
1. The project is implemented by python using the following libraries:
A. Jupyter Notebook
B. Tensorflow
2. The project includes Convolution Neural Network that can be used to extract the patterns.
3. This model first applies 1D convolution layers to get sensor datas
4. Max pooling is used to to downsample the data. T
5. Two LSTM layers :  capture temporal dependencies in the sequential data, with dropout added to prevent overfitting. The output of the LSTM is fed into dense layers for final feature transformation. 
6. A sigmoid-activated dense layer provides the binary classification output (arrhythmia detection).
   
## System Architecture:
![image](https://github.com/user-attachments/assets/d4b0e7c2-545f-40c4-8639-a4026746d3c5)



## Code
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# Load the dataset
csv_file = "/content/Car_Monitoring_Data.csv"
df = pd.read_csv(csv_file)

# Convert the Timestamp to a datetime object
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Data Preprocessing
features = ['Oxygen', 'CO2', 'CO', 'Methane', 'Temperature']
df['Alarm Triggered'] = df['Alarm Triggered'].astype(int)  # Convert boolean column to integer

# Normalize features
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
df_scaled['Timestamp'] = df['Timestamp']
df_scaled['Alarm Triggered'] = df['Alarm Triggered']

# Visualizations
plt.figure(figsize=(12, 6))
for feature in features:
    plt.plot(df_scaled['Timestamp'], df_scaled[feature], label=feature)
plt.title('Fluctuations of Features Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Normalized Values')
plt.legend()
plt.grid(True)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[features + ['Alarm Triggered']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Oxygen Level vs Time (Advanced Graph)
plt.figure(figsize=(14, 6))
sns.lineplot(x='Timestamp', y='Oxygen', data=df, label='Oxygen Level', color='blue')
plt.axhline(y=18, color='red', linestyle='--', label='Critical Level (18%)')
plt.title('Oxygen Level Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Oxygen Level (%)')
plt.legend()
plt.grid(True)
plt.show()

# Prediction using LSTM
SEQ_LENGTH = 10  # Sequence length for LSTM

# Create sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Prepare the data
oxygen_values = df['Oxygen'].values.reshape(-1, 1)
oxygen_normalized = scaler.fit_transform(oxygen_values)

X, y = create_sequences(oxygen_normalized, SEQ_LENGTH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Visualization of Predictions
plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, label="Actual Oxygen Levels", color='blue')
plt.plot(predictions, label="Predicted Oxygen Levels", color='orange', linestyle='--')
plt.title('LSTM Predictions vs Actual Oxygen Levels')
plt.xlabel('Test Sample Index')
plt.ylabel('Oxygen Level (%)')
plt.legend()
plt.grid(True)
plt.show()

```
## Output:
![image](https://github.com/user-attachments/assets/87a5c0d7-e3f0-4d8d-a559-13ad97a727c3)
![image](https://github.com/user-attachments/assets/bf187afe-ab7f-4f77-a438-3a1321c91412)
![image](https://github.com/user-attachments/assets/ed12b976-7a6e-4944-99cc-c767ee1f76a1)
![image](https://github.com/user-attachments/assets/5220a5f0-f1b4-4228-bf25-1dcab90555fb)





