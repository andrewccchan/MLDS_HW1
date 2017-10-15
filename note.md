## MLDS HW1 Log

### Model 1
RNN with LSTM.

#### Setting 1
- RNN with LSTM cells.
- Truncated backpropagation with `num_steps = 10`.
- Loss after 8 epochs ~ 1.3
- Kaggle score = 54
- output file: 01_linux.out

#### Setting 2
- Single layer LSTM cell
- Truncated backpropagation with `num_steps = 10`.
- **Boundary-aware batch generation**
- Loss after 50 epochs ~ 1.0
- Kaggle score = 29
- output file: 04\_phone\_{wise, sequence}.out
