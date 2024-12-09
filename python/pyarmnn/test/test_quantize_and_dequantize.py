# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
import pytest
import numpy as np

import pyarmnn as ann

# import generated so we can test for Dequantize_* and Quantize_*
# functions not available in the public API.
import pyarmnn._generated.pyarmnn as gen_ann


@pytest.mark.parametrize('method', ['Quantize_int8_t',
                                    'Quantize_uint8_t',
                                    'Quantize_int16_t',
                                    'Quantize_int32_t',
                                    'Dequantize_int8_t',
                                    'Dequantize_uint8_t',
                                    'Dequantize_int16_t',
                                    'Dequantize_int32_t'])
def test_quantize_exists(method):
    assert method in dir(gen_ann) and callable(getattr(gen_ann, method))


@pytest.mark.parametrize('dt, min, max', [('uint8', 0, 255),
                                          ('int8', -128, 127),
                                          ('int16', -32768, 32767),
                                          ('int32', -2147483648, 2147483647)])
def test_quantize_uint8_output(dt, min, max):
    result = ann.quantize(3.3274056911468506, 0.02620004490017891, 128, dt)
    assert type(result) is int and min <= result <= max


@pytest.mark.parametrize('dt', ['uint8',
                                'int8',
                                'int16',
                                'int32'])
def test_dequantize_uint8_output(dt):
    result = ann.dequantize(3, 0.02620004490017891, 128, dt)
    assert type(result) is float


def test_quantize_unsupported_dtype():
    with pytest.raises(ValueError) as err:
        ann.quantize(3.3274056911468506, 0.02620004490017891, 128, 'uint16')

    assert 'Unexpected target datatype uint16 given.' in str(err.value)


def test_dequantize_unsupported_dtype():
    with pytest.raises(ValueError) as err:
        ann.dequantize(3, 0.02620004490017891, 128, 'uint16')

    assert 'Unexpected value datatype uint16 given.' in str(err.value)


def test_dequantize_value_range():
    with pytest.raises(ValueError) as err:
        ann.dequantize(-1, 0.02620004490017891, 128, 'uint8')

    assert 'Value is not within range of the given datatype uint8' in str(err.value)


@pytest.mark.parametrize('dt, data', [('uint8', np.uint8(255)),
                                      ('int8',  np.int8(127)),
                                      ('int16', np.int16(32767)),
                                      ('int32', np.int32(2147483647)),

                                      ('uint8', np.int8(127)),
                                      ('uint8', np.int16(255)),
                                      ('uint8', np.int32(255)),

                                      ('int8', np.uint8(127)),
                                      ('int8', np.int16(127)),
                                      ('int8', np.int32(127)),

                                      ('int16', np.int8(127)),
                                      ('int16', np.uint8(255)),
                                      ('int16', np.int32(32767)),

                                      ('int32', np.uint8(255)),
                                      ('int16', np.int8(127)),
                                      ('int32', np.int16(32767))

                                      ])
def test_dequantize_numpy_dt(dt, data):
    result = ann.dequantize(data, 1, 0, dt)

    assert type(result) is float

    assert np.float32(data) == result
