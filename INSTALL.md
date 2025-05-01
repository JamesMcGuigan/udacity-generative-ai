# Python Install

## Anaconda
```
brew uninstall python3 pipx
brew install --cask anaconda

which python3  # python3 is /opt/homebrew/anaconda3/bin/python3
pip install torch
  
```

## Miniconda
- https://stackoverflow.com/a/60902863/748503
- https://developer.apple.com/metal/pytorch/
- https://discuss.pytorch.org/t/torch-not-compiled-with-cuda-enabled/112467
```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/.miniconda
pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorflow tensorflow-macos tensorflow-metal
which python3  # python3 is ~/.miniconda/bin/python3
```
```
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
```