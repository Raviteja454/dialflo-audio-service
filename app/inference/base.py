from abc import ABC, abstractmethod

import numpy as np


class AudioClassifier(ABC):

    @abstractmethod
    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict:
        pass
