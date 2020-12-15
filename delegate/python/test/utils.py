# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import tflite_runtime.interpreter as tflite
import numpy as np
import os


def run_mock_model(delegate, test_data_folder):
    model_path = os.path.join(test_data_folder, 'mock_model.tflite')
    interpreter = tflite.Interpreter(model_path=model_path,
                                     experimental_delegates=[delegate])
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()