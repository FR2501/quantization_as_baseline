from abc import abstractmethod, ABC
from typing import Any

import numpy as np
from padasip.filters import FilterNLMS

from quantizers import Quantizer

class Predictor(ABC):

    def __init__(self, quantizer: Quantizer):
        self._quantizer = quantizer

    def predict(self, input_path, output_path):
        input_data = self._predict(np.load(input_path)).astype(self._quantizer._quantized_dtype, casting="safe")
        if input_data is None:
            raise RuntimeError("Data could not be loaded.")
        print(input_data.dtype)

        np.save(output_path, input_data)

    def reconstruct(self, input_path, output_path):
        print(np.load(input_path).dtype)

        data = self._reconstruct(np.load(input_path)).astype(self._quantizer._original_dtype, casting="safe")
        if data is None:
            raise RuntimeError("Data could not be loaded.")

        print(f"saving to {output_path}")
        np.save(output_path, data)

    def get_error_bound(self):
        return self._quantizer.get_error_bound()
    
    @abstractmethod
    def get_unique_identifier(self) -> str:
        pass

    @abstractmethod
    def _predict(self, input_data) -> np.ndarray:
        pass

    @abstractmethod
    def _reconstruct(self, input_data) -> np.ndarray:
        pass


class NLMSPredictor(Predictor):

    def __init__(self, quantizer, window_size=32, mu=0.5, eps=0.01) -> None:
        super().__init__(quantizer)

        self.__window_size = window_size
        self.__mu = mu
        self.__eps = eps

    def get_unique_identifier(self) -> str:
        return f"nlms-{self.__window_size}"

    def _predict(self, input_data):
        print("predict")
        orig_shape = input_data.shape
        input_data = input_data.reshape(-1)

        nlms = FilterNLMS(self.__window_size, mu=self.__mu, eps=self.__eps, w="zeros")

        deltas = np.full(input_data.shape, 0, dtype=self._quantizer._quantized_dtype)
        reconstruction = np.full(input_data.shape, 0, dtype=self._quantizer._original_dtype)

        for i in range(len(input_data)):
            if i < self.__window_size + 1:
                deltas[i] = self._quantizer.quantize_single_value(input_data[i])
                reconstruction[i] = self._quantizer.unquantize_single_value(deltas[i])
            else:
                nlms.adapt(reconstruction[i-1], reconstruction[(i-1)-self.__window_size:(i-1)])
                prediction = nlms.predict(reconstruction[(i-1)-self.__window_size:(i-1)])

                deltas[i] = self._quantizer.quantize_single_value(input_data[i] - prediction)
                reconstruction[i] = prediction + self._quantizer.unquantize_single_value(deltas[i])
            if np.isnan(reconstruction[i]):
                print(reconstruction[i-50:i])
                raise RuntimeError()
        print(np.max(np.abs(reconstruction - input_data)))

        print(deltas.dtype)
        return deltas.reshape(orig_shape)
    
    def _reconstruct(self, input_data):
        print("reconstruct")
        orig_shape = input_data.shape
        deltas = input_data.reshape(-1)
        print(input_data.dtype, deltas.dtype)

        reconstruction = np.full(deltas.shape, np.nan, dtype=self._quantizer._original_dtype)

        nlms = FilterNLMS(self.__window_size, mu=self.__mu, eps=self.__eps, w="zeros")

        for i in range(len(reconstruction)):
            if i < self.__window_size + 1:
                reconstruction[i] = self._quantizer.unquantize_single_value(deltas[i])
            else:
                nlms.adapt(reconstruction[i-1], reconstruction[(i-1)-self.__window_size:i-1])
                prediction = nlms.predict(reconstruction[(i-1)-self.__window_size:(i-1)])

                reconstruction[i] = prediction + self._quantizer.unquantize_single_value(deltas[i])

        print(reconstruction.dtype)
        return reconstruction.reshape(orig_shape)
    

class SimpleLorenzoPredictor(Predictor):

    def _predict(self, input_data) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        orig_shape = input_data.shape
        input_data = input_data.reshape(-1)

        deltas = np.full(input_data.shape, np.nan, dtype=self._quantizer._quantized_dtype)
        reconstruction = np.full(input_data.shape, np.nan, dtype=input_data.dtype)

        deltas[0] = self._quantizer.quantize_single_value(input_data[0])
        reconstruction[0] = self._quantizer.unquantize_single_value(deltas[0])

        for i in range(1, len(input_data)):
            prediction = reconstruction[i-1]
            deltas[i] = self._quantizer.quantize_single_value(input_data[i] - prediction)
            reconstruction[i] = prediction + self._quantizer.unquantize_single_value(deltas[i])

        return deltas.reshape(orig_shape)

    
    def _reconstruct(self, input_data) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        orig_shape = input_data.shape
        deltas = input_data.reshape(-1)

        reconstruction = np.full(deltas.shape, np.nan, dtype=self._quantizer._original_dtype)

        reconstruction[0] = self._quantizer.unquantize_single_value(deltas[0])

        for i in range(1, len(reconstruction)):
            prediction = reconstruction[i-1]

            reconstruction[i] = prediction + self._quantizer.unquantize_single_value(deltas[i])

        return reconstruction.reshape(orig_shape)
    
    def get_unique_identifier(self) -> str:
        return "lorenzo"
    

class ZeroPredictor(Predictor):

    def _predict(self, input_data) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        orig_shape = input_data.shape
        input_data = input_data.reshape(-1)

        deltas = np.full(input_data.shape, 0, dtype=self._quantizer._quantized_dtype)

        for i in range(len(input_data)):
            prediction = 0
            deltas[i] = self._quantizer.quantize_single_value(input_data[i] - prediction)

        return deltas.reshape(orig_shape)

    
    def _reconstruct(self, input_data) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        orig_shape = input_data.shape
        deltas = input_data.reshape(-1)

        reconstruction = np.full(deltas.shape, np.nan, dtype=self._quantizer._original_dtype)

        for i in range(len(reconstruction)):
            prediction = 0

            reconstruction[i] = prediction + self._quantizer.unquantize_single_value(deltas[i])

        return reconstruction.reshape(orig_shape)

    def get_unique_identifier(self):
        return "quant"

