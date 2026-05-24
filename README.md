# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
The objective of this experiment is to develop a Recurrent Neural Network (RNN) model to predict future stock prices using historical closing price data. The model learns patterns and trends from past stock market data and forecasts future prices. The dataset used contains stock closing prices collected over a period of time, which are preprocessed and normalized before training the RNN model.


## DESIGN STEPS
### STEP 1: 
Load and normalize data, create sequences.

### STEP 2: 
Convert data to tensors and set up DataLoader.

### STEP 3: 
Define the RNN model architecture.

### STEP 4: 
Summarize, compile with loss and optimizer.

### STEP 5: 
Train the model with loss tracking.

### STEP 6: 
Predict on test data, plot actual vs. predicted prices.

## PROGRAM

### Name: MUKITHA V M

### Register Number: 212223040119

```
# Define RNN Model
    class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    train_losses = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")
    return train_losses


```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="648" height="784" alt="Screenshot 2026-05-24 222141" src="https://github.com/user-attachments/assets/3e1f65a5-0ac2-4c6b-86eb-61b038744e12" />


## True Stock Price, Predicted Stock Price vs time
<img width="904" height="540" alt="Screenshot 2026-05-24 222248" src="https://github.com/user-attachments/assets/f411a3f5-f730-424e-ae2f-ab20e104437f" />


### Predictions
<img width="284" height="39" alt="Screenshot 2026-05-24 222324" src="https://github.com/user-attachments/assets/d911923a-6c84-412d-bb51-a64f4e7303ae" />


## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
