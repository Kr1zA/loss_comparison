# Comparison of Loss Functions for Time Series Forecasting in NSSA field
Example code for paper Comparison of Loss Functions for Time Series Forecasting in NSSA field.

In the file draw_losses.ipynb, there are graphs of every loss function used.

In the file get_res.ipynb, there are the processed results.

In losses.py are definitions of all loss functions

All computations were done by script.py. Usage:
```
python script.py --cuda_device 0 --trial_loss mae
```
If you have multiple GPUs, you should choose one by --cuda_device.
To choose a loss function that will be used for training, set as a parameter after --cuda_device one of:
* mae
* mse
* huber
* angle_loss
* rmse
* nrmse
* rrmse
* msle
* rmsle
* mase
* rmsse
* poisson
* logCosh
* mape
* mbe
* rae
* rse
* kernelMSE
* quantile25
* quantile75

All results can be found here:
https://wandb.ai/kriza-upjs/loss_security

The dataset will be published in the future, and a link will be added here.