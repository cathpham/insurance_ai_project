import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

print(f"Numpy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")