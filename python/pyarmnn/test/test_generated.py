# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
import inspect
from typing import Tuple

import pytest

import pyarmnn._generated.pyarmnn as generated_armnn
import pyarmnn._generated.pyarmnn as generated_deserializer
import pyarmnn._generated.pyarmnn_onnxparser as generated_onnx
import pyarmnn._generated.pyarmnn_tfliteparser as generated_tflite

swig_independent_classes = ('IBackend',
                            'IDeviceSpec',
                            'IConnectableLayer',
                            'IInputSlot',
                            'IOutputSlot',
                            'IProfiler')


def get_classes(swig_independent_classes: Tuple):
    # We need to ignore some swig generated_armnn classes. This is because some are abstract classes
    # They cannot be created with the swig generated_armnn wrapper, therefore they don't need a destructor.
    # Swig also generates its own meta class - this needs to be ignored.
    ignored_class_names = (*swig_independent_classes, '_SwigNonDynamicMeta')
    return list(filter(lambda x: x[0] not in ignored_class_names,
                       inspect.getmembers(generated_armnn, inspect.isclass) +
                       inspect.getmembers(generated_deserializer, inspect.isclass) +
                       inspect.getmembers(generated_tflite, inspect.isclass) +
                       inspect.getmembers(generated_onnx, inspect.isclass)))


@pytest.mark.parametrize("class_instance", get_classes(swig_independent_classes), ids=lambda x: 'class={}'.format(x[0]))
class TestPyOwnedClasses:

    def test_destructors_exist_per_class(self, class_instance):
        assert getattr(class_instance[1], '__swig_destroy__', None)

    def test_owned(self, class_instance):
        assert getattr(class_instance[1], 'thisown', None)


@pytest.mark.parametrize("class_instance", swig_independent_classes)
class TestPyIndependentClasses:

    def test_destructors_does_not_exist_per_class(self, class_instance):
        assert not getattr(class_instance[1], '__swig_destroy__', None)

    def test_not_owned(self, class_instance):
        assert not getattr(class_instance[1], 'thisown', None)
