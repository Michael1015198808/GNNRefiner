import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
HIDDEN = 128
MODEL_DIR = 'models'
