from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.policies.policy import Policy
from src.utils.dataset import ImageClassificationDataset
from src.utils.image import resize_for_feature


@dataclass(frozen=True)
class EvaluationResult:
    """策略评估结果。"""

    score: float
    accuracy: float
    policy: Policy


class FeatureExtractor:
    """轻量图像特征提取器。"""

    def __init__(self, image_size: tuple[int, int] = (32, 32), histogram_bins: int = 16) -> None:
        self.image_size = image_size
        self.histogram_bins = histogram_bins

    def transform_one(self, image: np.ndarray) -> np.ndarray:
        """提取单张图像的统计特征。"""
        image = resize_for_feature(image, self.image_size)
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        channel_mean = bgr.reshape(-1, 3).mean(axis=0) / 255.0
        channel_std = bgr.reshape(-1, 3).std(axis=0) / 255.0
        histogram, _ = np.histogram(gray, bins=self.histogram_bins, range=(0, 255), density=True)
        histogram = histogram.astype(np.float32)
        return np.concatenate([channel_mean, channel_std, histogram]).astype(np.float32)

    def transform(self, images: tuple[np.ndarray, ...]) -> np.ndarray:
        """批量提取图像特征。"""
        return np.vstack([self.transform_one(image) for image in images])


class NearestCentroidClassifier:
    """最近质心分类器，用于低成本策略评分。"""

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "NearestCentroidClassifier":
        """训练每个类别的特征质心。"""
        if len(features) != len(labels):
            raise ValueError("特征数量必须与标签数量一致")
        classes = np.unique(labels)
        if len(classes) == 0:
            raise ValueError("训练标签不能为空")
        centroids = []
        for class_id in classes:
            class_features = features[labels == class_id]
            centroids.append(class_features.mean(axis=0))
        self.classes_ = classes
        self.centroids_ = np.vstack(centroids)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测每个样本最近的类别质心。"""
        if self.classes_ is None or self.centroids_ is None:
            raise RuntimeError("模型必须先训练再预测")
        distances = np.linalg.norm(features[:, None, :] - self.centroids_[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        return self.classes_[nearest]

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """计算分类准确率。"""
        predictions = self.predict(features)
        return float(np.mean(predictions == labels))


class LightweightPolicyEvaluator:
    """基于增强数据训练轻量模型并返回验证准确率。"""

    def __init__(
        self,
        train_dataset: ImageClassificationDataset,
        validation_dataset: ImageClassificationDataset,
        seed: int = 42,
        repeats: int = 1,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.seed = seed
        self.repeats = max(1, repeats)
        self.extractor = extractor or FeatureExtractor()

    def evaluate(self, policy: Policy) -> float:
        """执行策略到奖励的完整评估过程。"""
        scores = []
        validation_features = self.extractor.transform(self.validation_dataset.images)
        for repeat_index in range(self.repeats):
            augmented_train = self.train_dataset.augment(policy, seed=self.seed + repeat_index)
            train_features = self.extractor.transform(augmented_train.images)
            model = NearestCentroidClassifier().fit(train_features, augmented_train.labels)
            scores.append(model.score(validation_features, self.validation_dataset.labels))
        return float(np.mean(scores))

    def evaluate_result(self, policy: Policy) -> EvaluationResult:
        """返回包含策略和准确率的完整评估对象。"""
        accuracy = self.evaluate(policy)
        return EvaluationResult(score=accuracy, accuracy=accuracy, policy=policy)
