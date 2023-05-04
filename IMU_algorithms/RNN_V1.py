from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Assuming sensor_data and ground_truth_data are numpy arrays containing the data

# Scale the data
scaler = MinMaxScaler()
sensor_data_scaled = scaler.fit_transform(sensor_data)
ground_truth_data_scaled = scaler.fit_transform(ground_truth_data)

# Split the data into training and validation sets
train_size = int(0.8 * len(sensor_data_scaled))
train_sensor_data = sensor_data_scaled[:train_size, :]
train_ground_truth_data = ground_truth_data_scaled[:train_size, :]
val_sensor_data = sensor_data_scaled[train_size:, :]
val_ground_truth_data = ground_truth_data_scaled[train_size:, :]

# Define the input and output sequence lengths
input_seq_len = 10
output_seq_len = 1

# Create the input and target sequences


def create_sequences(data, seq_len):
    X = []
    y = []
    for i in range(len(data) - seq_len - 1):
        X.append(data[i:(i + seq_len), :])
        y.append(data[(i + seq_len), :])
    return np.array(X), np.array(y)


train_X, train_y = create_sequences(train_sensor_data, input_seq_len)
val_X, val_y = create_sequences(val_sensor_data, input_seq_len)

# Define the model architecture
model = Sequential()
model.add(LSTM(64, input_shape=(input_seq_len, sensor_data_scaled.shape[1])))
model.add(Dense(32, activation='relu'))
model.add(Dense(output_seq_len))

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model
history = model.fit(train_X, train_y, epochs=50,
                    batch_size=32, validation_data=(val_X, val_y))

# Evaluate the model
test_sensor_data_scaled = scaler.fit_transform(test_sensor_data)
test_ground_truth_data_scaled = scaler.fit_transform(test_ground_truth_data)
test_X, test_y = create_sequences(test_sensor_data_scaled, input_seq_len)
test_loss = model.evaluate(test_X, test_y)

# Make predictions
predictions = model.predict(test_X)
predictions_unscaled = scaler.inverse_transform(predictions)
