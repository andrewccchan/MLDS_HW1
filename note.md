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
- - output file: 07\_phone\_{wise, sequence}.out

## Model 2:
CNN plus LSTM

#### Observation 1
-	Overfitting occurs in this model
-	`cnn_filter_num` = `[32, 64, 64]`
-	`cnn_pool_size` = `[2, 1, 1]`
-	`fc_layer_size` = `[1024, 512, 256]`
-	`rnn_state_size` = `100`
-	`batch_size` = 128
-	@27epoch: Traning loss ~ 0.3 but validation ~ 0.4 to 0.5

#### Observations 2
- Overfitting.
-	`cnn_filter_num` = `[32, 32, 32]`
-	`cnn_pool_size` = `[2, 1, 1]`
-	`fc_layer_size` = `[1024, 512, 256]`
-	`rnn_state_size` = `100`
-	`batch_size` = 128
-	@25epoch: Traning loss ~ 0.4 but validation ~ 0.4 to 0.7

### Setting 1
-	`cnn_filter_num` = `[32, 32, 32]`
-	`cnn_pool_size` = `[2, 1, 1]`
-	`fc_layer_size` = `[640, 512, 256]`
-	`rnn_state_size` = `100`
-	`batch_size` = 128
-	`kernel` = 2D
-	`keep_prob` = 0.7 at training
-	@25 epoch: Traning loss ~ 0.7; validation ~ 0.5 to 0.7
-	@50 epoch: Tranining loss ~ 0.5 to 0.6, validation ~ 0.6 (slightly overfitted)
-	Kaggle Score: 11.3
- 	output file: 09\_phone\_{wise, sequence}.out