import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))

import torch
from torch import optim
from torch.utils import data

import candle

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')

#
#   Define data
#
N, D_in, H, D_out = 1750, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

dev_x = torch.randn(150, D_in)
dev_y = torch.randn(150, D_out)

test_x = torch.randn(200, D_in)
test_y = torch.randn(200, D_out)

#
#   Define model
#
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
#
#   Create dataloader from data
#
train_ds = data.TensorDataset(x, y)
dev_ds = data.TensorDataset(dev_x, dev_y)
test_ds = data.TensorDataset(test_x, test_y)

train_loader = data.DataLoader(train_ds, batch_size=10, shuffle=True)
dev_loader = data.DataLoader(dev_ds, batch_size=10, shuffle=False)
test_loader = data.DataLoader(test_ds, batch_size=10, shuffle=False)

#
#   Training
#
optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse = torch.nn.MSELoss()

callbacks = [
    candle.callbacks.AdaptiveLearningRateCallback(initial_learning_rate=0.01, change=0.1, num_epochs=1),
    candle.callbacks.ModelCheckpointCallback('model_checkpoints', after_num_epochs=1, after_num_batches=25)
]

trainer = candle.Trainer(model, optimizer,
                         targets=[candle.Target('MSE', mse)],
                         num_epochs=3,
                         use_cuda=False,
                         callbacks=callbacks)

train_log = trainer.train(train_loader, dev_loader)

#
#   Evaluation
#
eval_log = trainer.evaluate(test_loader)

#
#   Store results
#
logging.info('Store log data')
train_log.save_in_folder('train_log')
os.makedirs('train_loss', exist_ok=True)
os.makedirs('train_metrics', exist_ok=True)
train_log.save_loss_plots_at('train_loss')
train_log.save_metric_plots_at('train_metrics')

train_log.write_stats_to('training_result.txt')
eval_log.write_stats_to('evaluation_result.txt')
