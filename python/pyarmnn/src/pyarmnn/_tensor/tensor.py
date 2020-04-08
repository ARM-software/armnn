# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
"""
This file contains the custom python implementation for Arm NN Tensor objects.
"""
import numpy as np

from .._generated.pyarmnn import Tensor as annTensor, TensorInfo, DataType_QAsymmU8, DataType_QSymmS8, \
    DataType_QAsymmS8, DataType_Float32, DataType_QSymmS16, DataType_Signed32, DataType_Float16


class Tensor(annTensor):
    """Creates a PyArmNN Tensor object.

    This class overrides the swig generated Tensor class. The aim of
    this is to create an easy to use public api for the Tensor object.

    Memory is allocated and managed by this class, avoiding the need to manage
    a separate memory area for the tensor compared to the swig generated api.

    """

    def __init__(self, *args):
        """ Create Tensor object.

        Supported tensor data types:
            `DataType_QAsymmU8`,
            `DataType_QAsymmS8`,
            `DataType_QSymmS16`,
            `DataType_QSymmS8`,
            `DataType_Signed32`,
            `DataType_Float32`,
            `DataType_Float16`

        Examples:
            Create an empty tensor
            >>> import pyarmnn as ann
            >>> ann.Tensor()

            Create tensor given tensor information
            >>> ann.Tensor(ann.TensorInfo(...))

            Create tensor from another tensor i.e. copy a tensor
            >>> ann.Tensor(ann.Tensor())

        Args:
            tensor(Tensor, optional): Create Tensor from a Tensor i.e. copy.
            tensor_info (TensorInfo, optional): Tensor information.

        Raises:
            TypeError: unsupported input data type.
            ValueError: appropriate constructor could not be found with provided arguments.

        """
        self.__memory_area = None

        # TensorInfo as first argument, we need to create memory area manually
        if len(args) > 0 and isinstance(args[0], TensorInfo):
            self.__create_memory_area(args[0].GetDataType(), args[0].GetNumElements())
            super().__init__(args[0], self.__memory_area.data)

        # copy constructor - reference to memory area is passed from copied tensor
        # and armnn's copy constructor is called
        elif len(args) > 0 and isinstance(args[0], Tensor):
            self.__memory_area = args[0].get_memory_area()
            super().__init__(args[0])

        # empty constructor
        elif len(args) == 0:
            super().__init__()

        else:
            raise ValueError('Incorrect number of arguments or type of arguments provided to create Tensor.')

    def __copy__(self) -> 'Tensor':
        """ Make copy of a tensor.

        Make tensor copyable using the python copy operation.

        Note:
            The tensor memory area is NOT copied. Instead, the new tensor maintains a
            reference to the same memory area as the old tensor.

        Example:
            Copy empty tensor
            >>> from copy import copy
            >>> import pyarmnn as ann
            >>> tensor = ann.Tensor()
            >>> copied_tensor = copy(tensor)

        Returns:
            Tensor: a copy of the tensor object provided.

        """
        return Tensor(self)

    def __create_memory_area(self, data_type: int, num_elements: int):
        """ Create the memory area used by the tensor to output its results.

        Args:
            data_type (int): The type of data that will be stored in the memory area.
                             See DataType_*.
            num_elements (int): Determines the size of the memory area that will be created.

        """
        np_data_type_mapping = {DataType_QAsymmU8: np.uint8,
                                DataType_QAsymmS8: np.int8,
                                DataType_QSymmS8: np.int8,
                                DataType_Float32: np.float32,
                                DataType_QSymmS16: np.int16,
                                DataType_Signed32: np.int32,
                                DataType_Float16: np.float16}

        if data_type not in np_data_type_mapping:
            raise ValueError("The data type provided for this Tensor is not supported.")

        self.__memory_area = np.empty(shape=(num_elements,), dtype=np_data_type_mapping[data_type])

    def get_memory_area(self) -> np.ndarray:
        """ Get values that are stored by the tensor.

        Returns:
            ndarray : Tensor data (as numpy array).

        """
        return self.__memory_area
