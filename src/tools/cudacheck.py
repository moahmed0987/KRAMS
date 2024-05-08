import torch

# check if cuda is available
print("CUDA is available" if torch.cuda.is_available() else "CUDA is not available, using CPU")