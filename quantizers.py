import numpy as np

from abc import abstractmethod, ABC


class Quantizer(ABC):

    def __init__(self, original_dtype: np.dtype, quantized_dtype: np.dtype, error_bound: float) -> None:
        self._original_dtype = original_dtype
        self._quantized_dtype = quantized_dtype
        self._error_bound = error_bound

    def get_error_bound(self):
        return self._error_bound

    def quantize(self, input_path, output_path):
        original_array = np.load(input_path)

        if original_array.dtype != self._original_dtype:
            print(f"quantize() Warning: Different datatypes in quantizer. Expected {self._original_dtype}, found {original_array.dtype}.")

        print(output_path)
        np.save(output_path, self._quantize(original_array).astype(self._quantized_dtype, casting="safe"))

    def quantize_single_value(self, x):
        return self._quantize(x).astype(self._quantized_dtype, casting="safe")
    
    def unquantize_single_value(self, x):
        return self._unquantize(x).astype(self._original_dtype, casting="safe")
    
    def unquantize(self, input_path, output_path):
        quantized_array = np.load(input_path)

        if quantized_array.dtype != self._quantized_dtype:
            print(f"unquantize() Warning: Different datatypes in quantizer. Expected {self._quantized_dtype}, found {quantized_array.dtype}.")

        np.save(output_path, self._unquantize(quantized_array).astype(self._original_dtype, casting="safe"))

    @abstractmethod
    def _quantize(self, original_array: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _unquantize(self, quantized_array: np.ndarray) -> np.ndarray:
        pass


class LinearQuantizer(Quantizer):

    def __init__(self, original_dtype: np.dtype,  quantized_dtype: np.dtype, error_bound: float) -> None:
        super().__init__(original_dtype, quantized_dtype, error_bound)

        self.__bin_width = 1.98 * error_bound

    def _quantize(self, original_array: np.ndarray) -> np.ndarray:
        return ((original_array.astype(np.float128, casting="safe") + (0.5 * self.__bin_width)) // self.__bin_width).astype(self._quantized_dtype, casting="unsafe")

    def _unquantize(self, quantized_array: np.ndarray) -> np.ndarray:
        return (quantized_array.astype(np.float128, casting="safe") * self.__bin_width).astype(self._original_dtype, casting="unsafe")
