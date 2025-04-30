# Install

## Miniconda
https://stackoverflow.com/a/60902863/748503

```
brew uninstall python3 pipx
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda

which python3
pip install torch
```
[//]: # (source ~/miniconda/bin/activate)
[//]: # (# conda init zsh  # bash ~/.bash_conda)
[//]: # (conda update -n base -c defaults conda)
[//]: # (conda install conda-build)
[//]: # (conda install pytorch)
[//]: # ()
[//]: # (conda create -n synthesis python=3.9)
[//]: # (conda activate synthesis)
[//]: # (# conda remove --name metalearning2 --all)
