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

#### Setting 3
- Single layer LSTM cell
- Truncated backpropagation with `num_steps = 10`.
- Boundary-aware batch generation
- **Output smoothing**: length = 1 (suppress singleton)
- Loss after 50 epochs ~ 1.0
- Kaggle score = 16.24
- output file: 05\_phone\_{wise, sequence}.out

#### Setting 4
- Single layer LSTM cell
- Truncated backpropagation with `num_steps = 10`.
- Boundary-aware batch generation
- **Output smoothing**: length = 2
- Loss after 50 epochs ~ 1.0
- Kaggle score = 13.23
- output file: 06\_phone\_{wise, sequence}.out

#### Setting 5
- Single layer LSTM cell
- Truncated backpropagation with `num_steps = 10`.
- Boundary-aware batch generation
- **Output smoothing**: length = 2
- Loss after 50 epochs ~ 1.0
- Kaggle score = 14.7

### Setting 6
- Single layer LSTM cell
- Truncated backpropagation with `num_steps = 15`.
- Boundary-aware batch generation
- **Output smoothing**: length = 2
- Loss after 60 epochs ~ 0.9
- Kaggle score = 12.8
- - output file: 0\_phone\_{wise, sequence}.out