from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score


def binary_metrics(threshold: float = 0.5) -> MetricCollection:
    return MetricCollection({
        "accuracy": BinaryAccuracy(threshold=threshold),
        "precision": BinaryPrecision(threshold=threshold),
        "recall": BinaryRecall(threshold=threshold),
        "specificity": BinarySpecificity(threshold=threshold),
        "f1": BinaryF1Score(threshold=threshold),
        "auroc": BinaryAUROC(),
    })


def multiclass_metrics(num_classes: int) -> MetricCollection:
    return MetricCollection({
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
        "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
        "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
    })


def regression_metrics() -> MetricCollection:
    return MetricCollection({
        "mae": MeanAbsoluteError(),
        "rmse": MeanSquaredError(squared=False),
        "r2": R2Score(),
    })


def image_metrics(data_range: float = 1.0) -> MetricCollection:
    return MetricCollection({
        "mse": MeanSquaredError(),
        "psnr": PeakSignalNoiseRatio(data_range=data_range, dim=(1, 2, 3)),
        "ssim": StructuralSimilarityIndexMeasure(data_range=data_range),
    })
