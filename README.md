# candle

High level utility for training neural networks with pytorch.
Simple alternative to https://github.com/ncullen93/torchsample, but less functionality.

## Features

* Predefined train loop
* Callbacks (e.g. for output logging)
* Multiple Targets
* Metrics (e.g. for keeping track of accuracy)
* Log (automatic store of loss, metrics)
* Produce charts of losses and metrics
* Save results in files


## Installation

**Install PyTorch**  
See http://pytorch.org/

**Install candle**
```sh
pip install -e .
```