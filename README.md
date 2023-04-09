## Focal Loss Implementation

[source](https://arxiv.org/abs/1708.02002)

Yet another implementation of Focal Loss. In the `loss.py` file you can find:
- `FocalLoss` - original implementation of Focal Loss
- `FocalStarLoss` - implementation of Focal Loss * (see the original paper's appendix for details)
- `FocalSmoothLoss` - implementation of Focal Loss with smoothing

Also there is a notebook `focal.ipynb` with POC and some visualization. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/podidiving/FocallLossTorch/blob/main/focal.ipynb)

*Note* Tested with `torch>=2.0.0` but I believe that it should work fine with older versions (I just didn't try it)