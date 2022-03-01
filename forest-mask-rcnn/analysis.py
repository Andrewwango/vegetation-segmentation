from datasets import FreiburgDataset
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import copy, csv
from deeplabmodel import DeepLabv3Model
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/log.csv')
df.plot(x='epoch',figsize=(15,8))

