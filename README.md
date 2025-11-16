# Gold Price Direction with LSTM

This project uses a Long Short-Term Memory (LSTM) neural network to tell if the gold price (XAUUSD) will move **up or down 15 minutes from now**. The whole workflow lives in `Gold_LSTM.ipynb`.

## Files in this folder

- `Gold_LSTM.ipynb` – notebook with code and plots.
- `data/` – optional local CSVs.
- `best_lstm_model.pth` – model with the lowest validation loss.
- `best_lstm_model_acc.pth` – model with the best validation accuracy.
- `best_loss.txt`, `best_acc.txt` – text files that store those two numbers.
- `loss_acc_graph_output.png` – saved plot of training curves.
- `requirements.txt` – needed Python packages. (exclude pytorch install by yourself)

## How the notebook works

### 1. Get the data
- The first cell downloads `M1_Y3.csv` directly from GitHub:  
  `https://raw.githubusercontent.com/faiisu/DL_LSTM_XAUUSD_predict/main/data/M1_Y3.csv`
- Needed columns: `time`, `Open`, `High`, `Low`, `Close`, `TickVolume`.

### 2. Clean the table
- Remove the `real_volume` column (it is all zeros).
- Convert `time` to datetime, sort by time, and drop rows with missing values.
- Keep only the columns we use and reset the index. This makes later windows easy to build.

### 3. Create helper features
All new features are built with pandas and NumPy:
- **Returns & momentum**: differences of closing price (1, 3, 5 steps), log return, price change (Close − Open).
- **Moving averages**: EMA with span 5, 9, and 20 plus the distance between price and each EMA.
- **Volatility**: rolling standard deviation of log returns (10 and 20 steps), Average True Range over 5 steps, and price range (High − Low).
- **Volume**: 5-step moving average of `TickVolume` and 1-step change in volume.
- **Time features**: hour, minute, and sine/cosine pairs so the model knows that 23:59 is close to 00:00.
- After feature work, rows that contain NaN are dropped and the `time` column itself is removed (the useful info is already encoded).

### 4. Windowing and labeling
- Window size = **120 minutes** of past data.
- Prediction horizon = **15 minutes** into the future.
- For each window, look at the Close price 15 minutes ahead:
  - Label `1` (up) if the future Close is higher than the last Close inside the window.
  - Label `0` (down) otherwise.
- Every sample now has shape `(120, number_of_features)` and one label.

### 5. Split, balance, and scale
- Use `train_test_split` with `shuffle=False` to respect time order:
  - 50% train, 25% validation, 25% test (created through a two-step split).
- Balance the train set using undersampling so both labels appear the same number of times.
- Fit `StandardScaler` on **train** windows only (flattened to 2D) to avoid leaking test info. Then transform validation and test windows with the same scaler.
- Convert everything to PyTorch tensors and create `DataLoader`s with `batch_size = 64`. Only the train loader shuffles.

## LSTM model

```python
class LSTM_Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
        )
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc2(F.relu(self.fc1(last)))
```

- Loss: `CrossEntropyLoss`.
- Optimizer: `Adam`, learning rate `1e-4`.
- Epochs: 5 (you can raise this if you want more training).
- Device: CUDA if available, else CPU.

## Training loop

1. Read past values in batches from the train loader.
2. Run the batch through the LSTM, compute loss, do backprop, update weights.
3. After each epoch, check validation loss and accuracy.
4. If `train_acc > 0.51` and the validation score is better than before, save the model:
   - `best_lstm_model.pth` keeps the weights with the **lowest validation loss**.
   - `best_lstm_model_acc.pth` keeps the weights with the **highest validation accuracy**.
   - both model don't upload to github (model too large, over 100mb)
   - `best_loss.txt` and `best_acc.txt` store those numbers as simple text.
5. Show plots of loss and accuracy so you can see training progress live.

## Test results

The last cell reloads `best_lstm_model_acc.pth`, runs it on the untouched test set (24,876 samples), and prints the classification report:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| down  | 0.46      | 0.33   | 0.38 | 11,215 |
| up    | 0.55      | 0.67   | 0.61 | 13,661 |
| **Overall accuracy** | | | **0.52** | 24,876 |

The model is still better at calling “up” than “down”, even with balancing.

## How to run it yourself

1. (Optional) `python -m venv .venv` and activate it.
2. `pip install -r requirements.txt`
3. install `pytorch` that work with your GPU version
3. Open `Gold_LSTM.ipynb` in Jupyter, VS Code, or Google Colab.
4. Run cells from top to bottom. Keep the internet on for the CSV download.
5. After training, run the final “Test” cell to see the report with your latest weights.

## Ideas to improve

1. Train longer or tweak learning rate / hidden size.
2. Try weighted loss or focal loss so missed “down” moves matter more.
3. Use wider windows or add outside data (e.g., dollar index, macro news).
4. Experiment with bidirectional LSTM layers or attention.
5. Turn the saved `.pth` files into a simple script that listens to live candles.
