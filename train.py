from boundml.ml import train
import sys

import torch

devices = [d for d in range(torch.cuda.device_count())]
print([torch.cuda.get_device_name(d) for d in devices])

policy = train(sys.argv[1], 0.005, 2000, sys.argv[2], patience=5, reduce_factor=0.2, n_epochs_without_improvement=20)
