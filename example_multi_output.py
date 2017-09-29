import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))

import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils import data

import candle

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')


#
#   Data loader
#
class Dataset(data.Dataset):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, item):
        return self.x[item], self.y[item], self.z[item]


#
#   Define data
#
N, D_in, H, D_out = 1750, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
z = torch.randn(N, D_out)

dev_x = torch.randn(150, D_in)
dev_y = torch.randn(150, D_out)
dev_z = torch.randn(150, D_out)

test_x = torch.randn(200, D_in)
test_y = torch.randn(200, D_out)
test_z = torch.randn(200, D_out)


#
#   Define model
#
class Net(nn.Module):
    def __init__(self, d_in, d_out):
        super(Net, self).__init__()

        self.lin1 = nn.Linear(d_in, d_out)
        self.lin2 = nn.Linear(d_out, d_out)

    def forward(self, x):
        out1 = self.lin1(x)
        out2 = self.lin2(out1)

        return out1, out2


model = Net(D_in, D_out)

#
#   Create dataloader from data
#
train_ds = Dataset(x, y, z)
dev_ds = Dataset(dev_x, dev_y, dev_z)
test_ds = Dataset(test_x, test_y, test_z)

train_loader = data.DataLoader(train_ds, batch_size=10, shuffle=True)
dev_loader = data.DataLoader(dev_ds, batch_size=10, shuffle=False)
test_loader = data.DataLoader(test_ds, batch_size=10, shuffle=False)


#
#   Custom Metric
#

class MSEMetric(candle.Metric):
    def compute(self, output, target, model=None):
        return torch.pow(output - target, 2).mean()

    def cumulate(self, metric_values=[]):
        return np.array(metric_values).mean()

    @classmethod
    def columns(cls):
        return ['error']


#
#   Training
#
optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse = torch.nn.MSELoss()
mse2 = torch.nn.MSELoss()

targets = [
    candle.Target("mse_out1", mse, output_index=0, target_index=1),
    candle.Target("mse_out2", mse2, output_index=1, target_index=2)
]

metrics = [
    MSEMetric("mse_metric_out1", output_index=0, target_index=1),
    MSEMetric("mse_metric_out2", output_index=1, target_index=2)
]

trainer = candle.Trainer(model, optimizer,
                         targets=targets,
                         metrics=metrics,
                         num_epochs=3,
                         use_cuda=False)

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
