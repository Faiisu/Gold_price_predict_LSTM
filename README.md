# Gold Price LSTM Study

This project explores how an LSTM model behaves on a real-world gold price time series from data collection through model evaluation. The aim is to understand the inner workings of LSTM blocks, how to stabilize them (sequence length, gradients, and imbalance handling), and how to deploy them responsibly on sequential forecasting problems.

## Data Pipeline
- **Source**: 1-minute XAUUSD data (`M1_Y3.csv`) loaded directly from the GitHub dataset used in the notebook.
- **Cleaning**: removed the all-zero `real_volume` column, converted timestamps, sorted chronologically, and dropped missing rows.
- **Feature engineering**: 
  - Returns & momentum features (price change, multiple rolling returns, log returns)
  - EMA-based signals (EMA 5/9/20, differences from price)
  - Volatility signals (rolling std, ATR, price ranges)
  - Volume signals (moving averages and first differences of tick volume)
  - Time encodings (hour/minute plus sine/cosine cycles)
- **Windowing & labeling**: built sliding windows of 420 minutes with a 60-minute prediction horizon. Labels are `1` when the future close price is higher than the last close in the window and `0` otherwise.
- **Scaling**: fit a `StandardScaler` on the flattened training windows and reused it for validation/test tensors to keep leakage under control.

## Model & Training
- **Architecture**: 4-layer `nn.LSTM` (hidden size 128, batch-first) followed by a fully connected projection into the two classes. The final `softmax` is implicit inside `CrossEntropyLoss`.
- **Data loaders**: windows split chronologically into train/validation/test (50%/25%/25%) and batched with `DataLoader` (`batch_size=64`, shuffling train only).
- **Optimization**: Adam (`lr=1e-5`), `CrossEntropyLoss`, 15 epochs, and a checkpoint of the best validation score saved as `best_lstm_model.pth`.
- **Stability work**: monitored gradient behavior, sequence length, and class balance counts to ensure the LSTM did not collapse to a constant prediction.

## Evaluation & Result
After balancing the training batches, the model was run against roughly one year of gold price data (the held-out test set). Metrics from the confusion matrix and the classification report show that the model is meaningfully better than a naive “always up” rule:

- Accuracy: 0.60  
- Precision (down): 0.02  
- Precision (up): 0.60

The wide precision gap reveals that bullish calls dominate the skill, while bearish calls remain unreliable. These insights point to the next round of feature engineering and sampling strategies needed to improve downside detection.
