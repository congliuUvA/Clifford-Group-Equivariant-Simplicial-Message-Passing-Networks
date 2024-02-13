import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")


USE_DISTRIBUTED = "RANK" in os.environ
if USE_DISTRIBUTED:
    import torch.distributed as dist


def detach_and_cast(input, device, detach=True):
    if isinstance(input, torch.Tensor):
        if device is not None:
            input = input.to(device)
        if detach:
            input = input.detach()
        return input
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, list):
        keys = range(len(input))
    elif isinstance(input, dict):
        keys = input.keys()
    else:
        raise ValueError(f"Unknown input type {type(input)}.")
    for k in keys:
        input[k] = detach_and_cast(input[k], device)
    return input


def gather(input):
    if isinstance(input, torch.Tensor):
        # global_input = torch.zeros(
        #     dist.get_world_size() * len(input),
        #     *input.shape[1:],
        #     device=input.device,
        #     dtype=input.dtype,
        # )
        global_input = [torch.empty_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(global_input, input.contiguous())
        # dist.all_gather_into_tensor(global_input, input)
        # return global_input
        return torch.cat(global_input)

    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, list):
        keys = range(len(input))
    elif isinstance(input, dict):
        keys = input.keys()
    else:
        raise ValueError(f"Unknown input type {type(input)}.")
    for k in keys:
        input[k] = gather(input[k])
    return input


def all_gather(compute):
    def wrapper(metric):
        if not USE_DISTRIBUTED:
            return compute(metric)
        metric.collection = gather(metric.collection)
        return compute(metric)

    return wrapper


class Metric:
    def __init__(self, to_cpu=False):
        self.to_cpu = to_cpu
        self.collection = []

    def empty(self):
        return len(self.collection) == 0

    def update(self, input):
        self.collection.append(detach_and_cast(input, "cpu" if self.to_cpu else None))

    def compute(self):
        raise NotImplementedError

    def reset(self):
        self.collection.clear()


class MetricCollection:
    def __init__(self, metrics):
        self.metrics = metrics

    def empty(self):
        return all(metric.empty() for metric in self.metrics.values())

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.metrics:
                raise ValueError(f"Unknown metric {k}. Did you add it to the model metrics?")
            self.metrics[k].update(v)
        # for name, metric in self.metrics.items():
        #     metric.update(kwargs[name])

    def compute(self):
        # return {name: metric.compute() for name, metric in self.metrics.items()}
        result = {}
        for name, metric in self.metrics.items():
            if metric.empty():
                warnings.warn(f"Metric {name} is empty.")
                continue
            values = metric.compute()
            if isinstance(values, torch.Tensor):
                result[name] = values
            elif isinstance(values, dict):
                result.update(values)
            else:
                raise ValueError(f"Unknown return type {type(values)}.")

        return result

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def keys(self):
        return self.metrics.keys()

    def __repr__(self) -> str:
        return f"MetricCollection({self.metrics})"


class Accuracy(Metric):
    @all_gather
    def compute(self):
        cat = torch.cat(self.collection)
        return cat.sum(dim=0) / cat.numel()


class Loss(Metric):
    @all_gather
    def compute(self):
        return torch.mean(torch.cat(self.collection), dim=0)


class RMSRE(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @all_gather
    def compute(self):
        return torch.sqrt(torch.mean(torch.cat(self.collection), dim=0))


class RRMSE(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @all_gather
    def compute(self):
        error_squares, target_squares = zip(*self.collection)
        return torch.sqrt(
            torch.mean(torch.cat(error_squares), dim=0)
            / torch.mean(torch.cat(target_squares), dim=0)
        )


class BinaryAUROC(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _binary_clf_curve(
        preds,
        target,
        sample_weights=None,
        pos_label: int = 1,
    ):
        with torch.no_grad():
            desc_score_indices = torch.argsort(preds, descending=True)

            preds = preds[desc_score_indices]
            target = target[desc_score_indices]

            if sample_weights is not None:
                weight = sample_weights[desc_score_indices]
            else:
                weight = 1.0

            # pred typically has many tied values. Here we extract
            # the indices associated with the distinct values. We also
            # concatenate a value for the end of the curve.
            distinct_value_indices = torch.where(preds[1:] - preds[:-1])[0]
            threshold_idxs = F.pad(
                distinct_value_indices, [0, 1], value=target.size(0) - 1
            )
            target = (target == pos_label).to(torch.long)
            tps = torch.cumsum(target * weight, dim=0)[threshold_idxs]

            if sample_weights is not None:
                # express fps as a cumsum to ensure fps is increasing even in
                # the presence of floating point errors
                fps = torch.cumsum((1 - target) * weight, dim=0)[threshold_idxs]
            else:
                fps = 1 + threshold_idxs - tps

            return fps, tps, preds[threshold_idxs]

    def _binary_roc_compute(self, preds, target, pos_label: int = 1):
        fps, tps, thresholds = self._binary_clf_curve(
            preds=preds, target=target, pos_label=pos_label
        )
        # Add an extra threshold position to make sure that the curve starts at (0, 0)
        tps = torch.cat([torch.zeros(1, dtype=tps.dtype, device=tps.device), tps])
        fps = torch.cat([torch.zeros(1, dtype=fps.dtype, device=fps.device), fps])
        thresholds = torch.cat(
            [
                torch.ones(1, dtype=thresholds.dtype, device=thresholds.device),
                thresholds,
            ]
        )

        if fps[-1] <= 0:
            print(
                "No negative samples in targets, false positive value should be meaningless."
                " Returning zero tensor in false positive score",
            )
            fpr = torch.zeros_like(thresholds)
        else:
            fpr = fps / fps[-1]

        if tps[-1] <= 0:
            print(
                "No positive samples in targets, true positive value should be meaningless."
                " Returning zero tensor in true positive score",
            )
            tpr = torch.zeros_like(thresholds)
        else:
            tpr = tps / tps[-1]

        return fpr, tpr, thresholds

    @staticmethod
    def _auc_compute_without_check(x, y, direction: float, axis: int = -1):
        """Computes area under the curve using the trapezoidal rule.
        Assumes increasing or decreasing order of `x`.
        """
        with torch.no_grad():
            auc_ = torch.trapz(y, x, dim=axis) * direction
        return auc_

    def _binary_auroc_compute(self, preds, target, pos_label: int = 1):
        fpr, tpr, _ = self._binary_roc_compute(preds, target, pos_label)
        return self._auc_compute_without_check(fpr, tpr, 1.0)

    @all_gather
    def compute(self):
        preds, target = zip(*self.collection)
        preds = torch.cat(preds)
        target = torch.cat(target)
        return self._binary_auroc_compute(preds=preds, target=target, pos_label=1)


def build_roc(labels, score, t_eff=[0.3, 0.5]):
    if not isinstance(t_eff, list):
        t_eff = [t_eff]
    fpr, tpr, threshold = roc_curve(labels, score)
    idx = [np.argmin(np.abs(tpr - Eff)) for Eff in t_eff]
    eB, eS = fpr[idx], tpr[idx]
    return fpr, tpr, threshold, eB, eS


class LorentzMetric(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @all_gather
    def compute(self):
        preds, target = zip(*self.collection)
        preds = torch.cat(preds).cpu().numpy()
        target = torch.cat(target).cpu().numpy()
        assert preds.shape == target.shape
        assert preds.min() >= 0 and preds.max() <= 1

        fpr, tpr, threshold, eB, eS = build_roc(target, preds)

        auc = roc_auc_score(target, preds)

        return {"auc": auc, "eB_0.3": eB[0], "eB_0.5": eB[1]}

