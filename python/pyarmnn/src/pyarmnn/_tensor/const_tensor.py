# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
"""
This file contains the custom python implementation for Arm NN Const Tensor objects.
"""
import numpy as np

from .._generated.pyarmnn import DataType_QAsymmU8, DataType_QSymmS8, DataType_QSymmS16, DataType_Signed32, \
    DataType_QAsymmS8, DataType_Float32, DataType_Float16
from .._generated.pyarmnn import ConstTensor as AnnConstTensor, TensorInfo, Tensor


class ConstTensor(AnnConstTensor):
    """Creates a PyArmNN ConstTensor object.

    A ConstTensor is a Tensor with an immutable data store. Typically, a ConstTensor
    is used to input data into a network when running inference.

    This class overrides the swig generated Tensor class. The aim of
    this is to have an easy to use public API for the ConstTensor objects.

    """

    def __init__(self, *args):
        """
        Supported tensor data types:
            `DataType_QAsymmU8`,
            `DataType_QAsymmS8`,
            `DataType_QSymmS16`,
            `DataType_QSymmS8`,
            `DataType_Signed32`,
            `DataType_Float32`,
            `DataType_Float16`

        Examples:
            Create empty ConstTensor
            >>> import pyarmnn as ann
            >>> import numpy as np
            >>> ann.ConstTensor()

            Create ConstTensor given tensor info and input data
            >>> input_data = np.array(...)
            >>> ann.ConstTensor(ann.TensorInfo(...), input_data)

            Create ConstTensor from another ConstTensor i.e. copy ConstTensor
            >>> ann.ConstTensor(ann.ConstTensor())

            Create ConstTensor from tensor
            >>> ann.ConstTensor(ann.Tensor())

        Args:
            tensor (Tensor, optional): Create a ConstTensor from a Tensor.
            const_tensor (ConstTensor, optional): Create a ConstTensor from a ConstTensor i.e. copy.
            tensor_info (TensorInfo, optional): Tensor information.
            input_data (ndarray):   The numpy array will be transformed to a
                                    buffer according to type returned by `TensorInfo.GetDataType`.
                                    Input data values type must correspond to data type returned by
                                    `TensorInfo.GetDataType`.

        Raises:
            TypeError: Unsupported input data type.
            ValueError: Unsupported tensor data type, incorrect input data size and creation of ConstTensor from non-constant TensorInfo.
        """
        self.__memory_area = None

        # TensorInfo as first argument and numpy array as second
        if len(args) > 1 and isinstance(args[0], TensorInfo):
            if not isinstance(args[1], np.ndarray):
                raise TypeError('Data must be provided as a numpy array.')
            # if TensorInfo IsConstant is false
            elif not args[0].IsConstant():
                raise ValueError('TensorInfo when initializing ConstTensor must be set to constant.')
            else:
                self.__create_memory_area(args[0].GetDataType(), args[0].GetNumBytes(), args[0].GetNumElements(),
                                          args[1])
                super().__init__(args[0], self.__memory_area.data)

        # copy constructor - reference to memory area is passed from copied const
        # tensor and armnn's copy constructor is called
        elif len(args) > 0 and isinstance(args[0], (ConstTensor, Tensor)):
            # if TensorInfo IsConstant is false
            if not args[0].GetInfo().IsConstant():
                raise ValueError('TensorInfo of Tensor when initializing ConstTensor must be set to constant.')
            else:
                self.__memory_area = args[0].get_memory_area()
                super().__init__(args[0])

        # empty tensor
        elif len(args) == 0:
            super().__init__()

        else:
            raise ValueError('Incorrect number of arguments or type of arguments provided to create Const Tensor.')

    def __copy__(self) -> 'ConstTensor':
        """ Make copy of a const tensor.

        Make const tensor copyable using the python copy operation.

        Note:
            The tensor memory area is NOT copied. Instead, the new tensor maintains a
            reference to the same memory area as the old tensor.

        Example:
            Copy empty tensor
            >>> from copy import copy
            >>> import pyarmnn as ann
            >>> tensor = ann.ConstTensor()
            >>> copied_tensor = copy(tensor)

        Returns:
            Tensor: a copy of the tensor object provided.

        """
        return ConstTensor(self)

    @staticmethod
    def __check_size(data: np.ndarray, num_bytes: int, num_elements: int):
        """ Check the size of the input data against the number of bytes provided by tensor info.

        Args:
            data (ndarray): Input data.
            num_bytes (int): Number of bytes required by tensor info.
            num_elements: Number of elements required by tensor info.

        Raises:
            ValueError: number of bytes in input data does not match tensor info.

        """
        size_in_bytes = data.nbytes
        elements = data.size

        if size_in_bytes != num_bytes:
            raise ValueError(
                "ConstTensor requires {} bytes, {} provided. "
                "Is your input array data type ({}) aligned with TensorInfo?".format(num_bytes, size_in_bytes,
                                                                                     data.dtype))
        if elements != num_elements:
            raise ValueError("ConstTensor requires {} elements, {} provided.".format(num_elements, elements))

    def __create_memory_area(self, data_type: int, num_bytes: int, num_elements: int, data: np.ndarray):
        """ Create the memory area used by the tensor to output its results.

        Args:
            data_type (int): The type of data that will be stored in the memory area.
                             See DataType_*.
            num_bytes (int): Determines the size of the memory area that will be created.
            num_elements (int): Determines number of elements in memory area.
            data (ndarray): Input data as numpy array.

        """
        np_data_type_mapping = {DataType_QAsymmU8: np.uint8,
                                DataType_QAsymmS8: np.int8,
                                DataType_QSymmS8: np.int8,
                                DataType_Float32: np.float32,
                                DataType_QSymmS16: np.int16,
                                DataType_Signed32: np.int32,
                                DataType_Float16: np.float16}

        if data_type not in np_data_type_mapping:
            raise ValueError("The data type provided for this Tensor is not supported: {}".format(data_type))

        if np_data_type_mapping[data_type] != data.dtype:
            raise TypeError("Expected data to have type {} for type {} but instead got numpy.{}".format(np_data_type_mapping[data_type], data_type, data.dtype))

        self.__check_size(data, num_bytes, num_elements)

        self.__memory_area = data
        self.__memory_area.flags.writeable = False

    def get_memory_area(self) -> np.ndarray:
        """ Get values that are stored by the tensor.

        Returns:
             ndarray: Tensor data (as numpy array).

        """
        return self.__memory_area
