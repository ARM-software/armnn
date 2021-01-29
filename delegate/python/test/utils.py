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

def run_inference(test_data_folder, model_filename, inputs, delegates=None):
    model_path = os.path.join(test_data_folder, model_filename)
    interpreter = tflite.Interpreter(model_path=model_path,
                                     experimental_delegates=delegates)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set inputs to tensors.
    for i in range(len(inputs)):
        interpreter.set_tensor(input_details[i]['index'], inputs[i])

    interpreter.invoke()

    results = []
    for output in output_details:
        results.append(interpreter.get_tensor(output['index']))

    return results

def compare_outputs(outputs, expected_outputs):
    assert len(outputs) == len(expected_outputs), 'Incorrect number of outputs'
    for i in range(len(expected_outputs)):
        assert outputs[i].shape == expected_outputs[i].shape, 'Incorrect output shape on output#{}'.format(i)
        assert outputs[i].dtype == expected_outputs[i].dtype, 'Incorrect output data type on output#{}'.format(i)
        assert outputs[i].all() == expected_outputs[i].all(), 'Incorrect output value on output#{}'.format(i)