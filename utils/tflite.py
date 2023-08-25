from typing import Callable, List, Tuple

import numpy as np
import tensorflow as tf

class TfLiteWrapper:

  interpreter: tf.lite.Interpreter = None
  quantizers: List[Tuple[int, Callable[[np.ndarray], np.ndarray]]]
  dequantizers: List[Tuple[int, Callable[[np.ndarray], np.ndarray]]]

  def __init__(self, model_path: str):
    self.interpreter = tf.lite.Interpreter(
        model_path = model_path
    )
    self.interpreter.allocate_tensors()
    self.quantizers = [
        (input_detail['index'], lambda data: (data / input_detail['quantization'][0] + input_detail['quantization'][1]).astype(input_detail['dtype']) if input_detail['dtype'] == np.uint8 else data.astype(input_detail['dtype']))
        for input_detail in self.interpreter.get_input_details()
    ]
    self.dequantizers = [
        (output_detail['index'], lambda data: (data.astype(np.float32) - output_detail['quantization'][1]) * output_detail['quantization'][0] if output_detail['dtype'] == np.uint8 else data.astype(output_detail['dtype']))
        for output_detail in self.interpreter.get_output_details()
    ]

  def predict(self, data: List[np.ndarray]) -> List[np.ndarray]:
    for idx in range(len(self.quantizers)):
      index, quantizer = self.quantizers[idx]
      self.interpreter.set_tensor(index, np.expand_dims(quantizer(data[idx]), 0))
    self.interpreter.invoke()
    return [
        dequant(self.interpreter.get_tensor(index)[0]) for index, dequant in self.dequantizers
    ]
