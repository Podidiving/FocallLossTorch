"""
The following code contains different implementations of Focal Loss

Each module has a lot of in common. 
It may seem reasonable to rearrange the code to follow DRY principle.
But the idea is that you can use each module as is. Just copy appropriate loss
function in your code and try it out.
"""
import typing as tp

import torch
from torch.nn import functional as F


class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        restore_dimensions: bool = True,
    ):
        """reimplementation of FocalLoss
        source: https://arxiv.org/abs/1708.02002
        Args:
            gamma (float, optional): gamma parameter. Defaults to 2.0.
            alpha (float, optional): alpha parameter. Defaults to 0.25.
            restore_dimensions (bool, optional):
            whether to restore the dimensions
            if reduction is "none" and there are more than two dimensions.
            Defaults to True.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.restore_dimensions = restore_dimensions

    def reshape_tensor(self, x: torch.Tensor) -> torch.Tensor:
        B, C, *_ = x.shape
        x = x.view(B, C, -1)
        x = x.transpose(1, 2)
        x = x.reshape(-1, C).contiguous()
        return x

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: tp.Union[
            tp.Literal["none"], tp.Literal["mean"], tp.Literal["sum"]
        ] = "none",
    ) -> torch.Tensor:
        """Focall Loss forward pass implementation

        Args:
            logits (torch.Tensor): Output logits of your model.
            tensor of shape [N, C, d_1, ..., d_k], where
            d_1, ..., d_k are optional.
            targets (torch.Tensor):  Ground truth values.
            tensor of shape [N, d_1, ..., d_k], where
            each element is a number in [0,C) range.
            reduction (str, optional):
            What reduction should be applied. Available values are:
                - "none",
                - "mean",
                - "sum"
            Defaults to "none".

        Raises:
            ValueError:
            raised when reduction parameter is not among the acceptable values

        Returns:
            loss: result of FocallLoss
        """
        if reduction == "none":
            original_shape = logits.shape
        else:
            original_shape = None

        # reshape for multidimensional input
        if len(logits.shape) > 2:
            logits = self.reshape_tensor(logits)

        if len(targets.shape) > 2:
            targets = targets.view(-1)

        B, *_ = logits.shape

        ohe = torch.zeros_like(logits)
        ohe[torch.arange(B), targets] = 1.0

        p = torch.sigmoid(logits)
        pt = p * ohe + (1.0 - p) * (1.0 - ohe)

        bce = F.binary_cross_entropy_with_logits(
            logits, ohe, reduction="none"
        )

        loss = bce * ((1.0 - pt) ** self.gamma)

        if self.alpha > 0:
            alpha_t = self.alpha * ohe + (1.0 - self.alpha) * (1.0 - ohe)
            loss *= alpha_t

        if reduction == "none":
            if self.restore_dimensions:
                assert original_shape is not None
                if len(original_shape) > 2:
                    shape_ = (
                        [original_shape[0]]
                        + list(original_shape[2:])
                        + [original_shape[1]]
                    )
                    loss = loss.view(shape_)
                    loss = loss.permute(
                        *([0] + [-1] + [i for i in range(1, len(shape_) - 1)])
                    )
                    loss = loss.contiguous()
            return loss
        elif reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")


class FocalStarLoss(torch.nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        beta: float = 0.0,
        restore_dimensions: bool = True,
    ):
        """reimplementation of FocalLoss*
        source: https://arxiv.org/abs/1708.02002
        (it's described in the paper's appendix)
        Args:
            gamma (float, optional): gamma parameter. Defaults to 2.0.
            alpha (float, optional): alpha parameter. Defaults to 0.25.
            beta (float, optional): beta parameter. Defaults to 0.0.
            restore_dimensions (bool, optional):
            whether to restore the dimensions
            if reduction is "none" and there are more than two dimensions.
            Defaults to True.
        """
        super().__init__()
        assert gamma > 0, "gamma must be positive"
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.restore_dimensions = restore_dimensions

    def reshape_tensor(self, x: torch.Tensor) -> torch.Tensor:
        B, C, *_ = x.shape
        x = x.view(B, C, -1)
        x = x.transpose(1, 2)
        x = x.reshape(-1, C).contiguous()
        return x

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: tp.Union[
            tp.Literal["none"], tp.Literal["mean"], tp.Literal["sum"]
        ] = "none",
    ) -> torch.Tensor:
        """Focall Loss* forward pass implementation

        Args:
            logits (torch.Tensor): Output logits of your model.
            tensor of shape [N, C, d_1, ..., d_k], where
            d_1, ..., d_k are optional.
            targets (torch.Tensor):  Ground truth values.
            tensor of shape [N, d_1, ..., d_k], where
            each element is a number in [0,C) range.
            reduction (str, optional):
            What reduction should be applied. Available values are:
                - "none",
                - "mean",
                - "sum"
            Defaults to "none".

        Raises:
            ValueError:
            raised when reduction parameter is not among the acceptable values

        Returns:
            loss: result of FocallLoss
        """
        if reduction == "none":
            original_shape = logits.shape
        else:
            original_shape = None

        # reshape for multidimensional input
        if len(logits.shape) > 2:
            logits = self.reshape_tensor(logits)

        if len(targets.shape) > 2:
            targets = targets.view(-1)

        B, *_ = logits.shape

        ohe = torch.zeros_like(logits)
        ohe[torch.arange(B), targets] = 1.0

        xt = logits * (2 * ohe - 1)
        pt = self.gamma * (xt + self.beta)
        try:
            loss = -(F.logsigmoid(pt)) / self.gamma
        except NotImplementedError:
            # mps device doesn't have support for logsigmoid
            loss = -(torch.log(torch.sigmoid(pt) + 1e-9)) / self.gamma

        if self.alpha > 0:
            alpha_t = self.alpha * ohe + (1.0 - self.alpha) * (1.0 - ohe)
            loss *= alpha_t

        if reduction == "none":
            if self.restore_dimensions:
                assert original_shape is not None
                if len(original_shape) > 2:
                    shape_ = (
                        [original_shape[0]]
                        + list(original_shape[2:])
                        + [original_shape[1]]
                    )
                    loss = loss.view(shape_)
                    loss = loss.permute(
                        *([0] + [-1] + [i for i in range(1, len(shape_) - 1)])
                    )
                    loss = loss.contiguous()
            return loss
        elif reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")


class FocalSmoothLoss(torch.nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        smooth: tp.Optional[float] = None,
        restore_dimensions: bool = True,
    ):
        """reimplementation of FocalLoss with smoothing
        source: https://arxiv.org/abs/1708.02002
        Args:
            gamma (float, optional): gamma parameter. Defaults to 2.0.
            alpha (float, optional): alpha parameter. Defaults to 0.25.
            smooth (float, optional): smooth coefficient.
            If it is not None value and targets are passed
            as labels positive label will be calculated as 1.0 - smooth,
            negative as smooth.
            Defaults to None.
            restore_dimensions (bool, optional):
            whether to restore the dimensions
            if reduction is "none" and there are more than two dimensions.
            Defaults to True.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth
        self.restore_dimensions = restore_dimensions

    def reshape_tensor(self, x: torch.Tensor) -> torch.Tensor:
        B, C, *_ = x.shape
        x = x.view(B, C, -1)
        x = x.transpose(1, 2)
        x = x.reshape(-1, C).contiguous()
        return x

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: tp.Union[
            tp.Literal["none"], tp.Literal["mean"], tp.Literal["sum"]
        ] = "none",
    ) -> torch.Tensor:
        """Focall Loss forward pass with smoothing implementation

        Args:
            logits (torch.Tensor): Output logits of your model.
            tensor of shape [N, C, d_1, ..., d_k], where
            d_1, ..., d_k are optional.
            targets (torch.Tensor): Ground truth values.
            Can be passed as:
            - labels:
                tensor of shape [N, d_1, ..., d_k], where
                each element is a number in [0,C) range.
            - binary labels:
                tensor of shape [N, C, d_1, ..., d_k], where
                each element is a number in [0,1] range.
            reduction (str, optional):
            What reduction should be applied. Available values are:
                - "none",
                - "mean",
                - "sum"
            Defaults to "none".

        Raises:
            ValueError:
            - raised when reduction parameter is not among
            the acceptable values
            - raised when smooth is none and targets are passed as labels
        Returns:
            loss: result of FocallLoss
        """
        if reduction == "none":
            original_shape = logits.shape
        else:
            original_shape = None

        # build smoothed target values
        if targets.shape != logits.shape:
            if self.smooth is None:
                raise ValueError(
                    "smooth parameter is not set and"
                    " targets are passed as labels"
                )
            ohe = torch.full_like(logits, self.smooth)
            targets = targets.unsqueeze(1)
            ohe = ohe.scatter_(
                1,
                index=targets,
                src=torch.full_like(targets.to(logits), 1.0 - self.smooth),
            )
            targets = ohe

        # reshape for multidimensional input
        if len(logits.shape) > 2:
            logits = self.reshape_tensor(logits)

        if len(targets.shape) > 2:
            targets = self.reshape_tensor(targets)

        p = torch.sigmoid(logits)
        pt = p * targets + (1.0 - p) * (1.0 - targets)

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        loss = bce * ((1.0 - pt) ** self.gamma)

        if self.alpha > 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (
                1.0 - targets
            )
            loss *= alpha_t

        if reduction == "none":
            if self.restore_dimensions:
                assert original_shape is not None
                if len(original_shape) > 2:
                    shape_ = (
                        [original_shape[0]]
                        + list(original_shape[2:])
                        + [original_shape[1]]
                    )
                    loss = loss.view(shape_)
                    loss = loss.permute(
                        *([0] + [-1] + [i for i in range(1, len(shape_) - 1)])
                    )
                    loss = loss.contiguous()
            return loss
        elif reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")


if __name__ == "__main__":
    fl1 = FocalLoss()
    fl2 = FocalSmoothLoss(smooth=0.0)
    fsl = FocalStarLoss()
    inputs = torch.randn(100, 10)
    targets = torch.randint(0, 10, (100,))
    loss1 = fl1(inputs, targets, "mean")
    loss2 = fl2(inputs, targets, "mean")
    assert torch.allclose(loss1, loss2)

    inputs = torch.randn(100, 10, 12, 12)
    targets = torch.randint(0, 10, (100, 12, 12))

    l1 = fl1(inputs, targets, "none")
    l2 = fl2(inputs, targets, "none")
    l3 = fsl(inputs, targets, "none")
    assert l1.shape == inputs.shape
    assert l2.shape == inputs.shape
    assert l3.shape == inputs.shape

    print("Seems ok")
