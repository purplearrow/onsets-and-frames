import pathlib
pathlib.Path.ls = lambda x: list(x.iterdir())

import os
from datetime import datetime

import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate
from onsets_and_frames import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

leave_one_out=None
batch_size = 8
sequence_length = 327680
model_complexity = 48

if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
    batch_size //= 2
    sequence_length //= 2
    print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

train_groups, validation_groups = ['train'], ['validation', 'test']

if leave_one_out is not None:
    all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
    train_groups = list(all_years - {str(leave_one_out)})
    validation_groups = [str(leave_one_out)]

dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
dataset.purplearrow_process(groups=validation_groups, id=3)    
