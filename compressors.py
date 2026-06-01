from abc import ABC, abstractmethod
import os
import tempfile


class Compressor(ABC):

    @abstractmethod
    def get_error_bound(self) -> float:
        pass

    @abstractmethod
    def get_unique_identifier(self) -> str:
        pass

    def compress(self, input_path, output_path):
        self._compress(input_path, output_path)

    @abstractmethod
    def _compress(self, input_path, output_path) -> None:
        pass

    def decompress(self, input_path, output_path):
        self._decompress(input_path, output_path)

    @abstractmethod
    def _decompress(self, input_path, output_path) -> None:
        pass

    def __hash__(self) -> int:
        return hash(self.get_unique_identifier())
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, GenericPQECompressor):
            return False
        
        return self.get_unique_identifier() == value.get_unique_identifier()
    
    def __str__(self) -> str:
        return self.get_unique_identifier()
    
    def __lt__(self, other):
        if not isinstance(other, GenericPQECompressor):
            return False
        else:
            return self.get_unique_identifier() < other.get_unique_identifier()


class GenericPQECompressor(Compressor):
    def __init__(self, predictor, entropy_coder, prediction_output_path=None, name=None) -> None:
        super().__init__()

        self.__predictor = predictor
        self.__entropy_coder = entropy_coder
        self.__name = name
        self.__prediction_output_path = prediction_output_path

    def get_unique_identifier(self) -> str:
        if self.__name:
            return self.__name
        
        return f'genericpqe_{self.__predictor.get_unique_identifier()}_{self.__entropy_coder.get_unique_identifier()}'
    
    def get_error_bound(self):
        return self.__predictor.get_error_bound()
    
    def get_predictor(self):
        return self.__predictor
    
    def set_prediction_output_path(self, prediction_output_path):
        self.__prediction_output_path = prediction_output_path

    def _compress(self, input_path, output_path) -> None:
        if self.__prediction_output_path and not os.path.exists(self.__prediction_output_path):
            self.__predictor.predict(input_path, self.__prediction_output_path)
        if not self.__prediction_output_path:
            self.__prediction_output_path = tempfile.NamedTemporaryFile().name + ".npy"
        self.__entropy_coder.encode(self.__prediction_output_path, output_path)

    def _decompress(self, input_path, output_path) -> None:
        intermediate_path = tempfile.NamedTemporaryFile().name + ".npy"
        self.__entropy_coder.decode(input_path, intermediate_path)
        self.__predictor.reconstruct(intermediate_path, output_path)
