from abc import ABC, abstractmethod
import onnx

class BasePass(ABC):
    """
    Abstract base class for all optimization passes.
    Every pass receives an onnx.ModelProto and returns a modified one.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging and reports."""
        pass

    @abstractmethod
    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """
        Apply the optimization pass.
        Must return a valid ONNX model.
        Must not alter accuracy.
        """
        pass

    def __repr__(self):
        return f"<Pass: {self.name}>"
