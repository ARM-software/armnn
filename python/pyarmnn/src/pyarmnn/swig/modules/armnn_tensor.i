//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%{
#include "armnn/Tensor.hpp"
%}

%include <typemaps/tensor_memory.i>
%include <typemaps/tensor_shape.i>

namespace armnn
{

%feature("docstring",
"
Class for holding the shape information of an Arm NN tensor.

This class is iterable. You can iterate over it to get each value of the Tensor shape.

Examples:
    Obtain tensor shape information as a list.
    >>> import pyarmnn as ann
    >>> import numpy as np
    >>>
    >>> tensor_info = ann.TensorInfo(ann.TensorShape((4, 2, 1, 3)), ann.DataType_Float32)
    >>> tensor = ann.ConstTensor(tensor_info, np.ones([4, 2, 1, 3], dtype=np.float32))
    >>> print(list(tensor.GetShape()))
    [4, 2, 1, 3]

") TensorShape;
class TensorShape
{
    // Make TensorShape iterable so we can return shape dims easily.
    %pythoncode %{
    def __iter__(self):
        for dim in range(self.GetNumDimensions()):
            yield self[dim]
    %}


public:
    %tensor_shape_typemap(unsigned int numDimensions, const unsigned int* dimensionSizes);
    TensorShape(unsigned int numDimensions, const unsigned int* dimensionSizes);
    %clear_tensor_shape_typemap(unsigned int numDimensions, const unsigned int* dimensionSizes);

    %feature("docstring",
    "
    Returns the number of dimensions in this TensorShape.

    Returns:
        int: The number of dimensions in this TensorShape.

    ") GetNumDimensions;
    unsigned int GetNumDimensions() const;

    %feature("docstring",
    "
    Returns the total number of elements for a tensor with this TensorShape.

    Returns:
        int: The total number of elements for a tensor with this TensorShape.

    ") GetNumElements;
    unsigned int GetNumElements() const;

};

%extend TensorShape {

    unsigned int __getitem__(unsigned int i) const {
        return $self->operator[](i);
    }
    void __setitem__(unsigned int i, unsigned int val) {
         $self->operator[](i) = val;
    }

    std::string __str__() {
        std::string dim = "NumDimensions: " + std::to_string($self->GetNumDimensions());
        std::string elm = "NumElements: " + std::to_string($self->GetNumElements());

        std::string shapeStr = "TensorShape{Shape(";

        auto numDimensions = $self->GetNumDimensions();
        auto sizeDims = $self->GetNumDimensions();
         for (unsigned int i = 0; i < numDimensions; i++) {
            shapeStr += std::to_string($self->operator[](i));

            if (sizeDims - 1 > 0) {
                shapeStr += ", ";
            }
            sizeDims--;
            }
        shapeStr = shapeStr + "), " + dim + ", " + elm + "}";
        return shapeStr;
    }

}


%feature("docstring",
"
Class for holding the tensor information of an Arm NN tensor such as quantization, datatype, shape etc.

") TensorInfo;
class TensorInfo
{
public:
    TensorInfo();

    TensorInfo(const TensorInfo& other);

    TensorInfo(const TensorShape& shape, DataType dataType,
        float quantizationScale = 0.0f, int32_t quantizationOffset = 0,
        bool isConstant = False);

    %feature("docstring",
    "
    Get the tensor shape.

    Return:
        TensorShape: Current shape of the tensor.

    ") GetShape;
    TensorShape& GetShape();

    %feature("docstring",
    "
    Set the tensor shape. Must have the same number of elements as current tensor.

    Args:
        newShape (TensorShape): New tensor shape to reshape to.

    ") SetShape;
    void SetShape(const TensorShape& newShape);

    %feature("docstring",
    "
    Returns the number of dimensions in this Tensor.

    Returns:
        int: The number of dimensions in this Tensor.

    ") GetNumDimensions;
    unsigned int GetNumDimensions() const;

    %feature("docstring",
    "
    Returns the total number of elements for this Tensor.

    Returns:
        int: The total number of elements for this Tensor.

    ") GetNumElements;
    unsigned int GetNumElements() const;

    %feature("docstring",
    "
    Get the tensor datatype.

    Returns:
        DataType: Current tensor DataType.

    ") GetDataType;
    DataType GetDataType() const;

    %feature("docstring",
    "
    Set the tensor datatype.

    Args:
        type (DataType): DataType to set the tensor to.

    ") SetDataType;
    void SetDataType(DataType type);

    %feature("docstring",
    "
    Get the value of the tensors quantization scale.

    Returns:
        float: Tensor quantization scale value.

    ") GetQuantizationScale;
    float GetQuantizationScale() const;

    %feature("docstring",
    "
    Get the value of the tensors quantization offset.

    Returns:
        int: Tensor quantization offset value.

    ") GetQuantizationOffset;
    int32_t GetQuantizationOffset() const;

    %feature("docstring",
    "
    Set the value of the tensors quantization scale.

    Args:
        scale (float): Scale value to set.

    ") SetQuantizationScale;
    void SetQuantizationScale(float scale);

    %feature("docstring",
    "
    Set the value of the tensors quantization offset.

    Args:
        offset (int): Offset value to set.

    ") SetQuantizationOffset;
    void SetQuantizationOffset(int32_t offset);

    %feature("docstring",
    "
    Returns true if the tensor is a quantized data type.

    Returns:
        bool: True if the tensor is a quantized data type.

    ") IsQuantized;
    bool IsQuantized() const;

    %feature("docstring",
    "
    Returns true if the tensor info is constant.

    Returns:
        bool: True if the tensor info is constant.

    ") IsConstant;
    bool IsConstant() const;

    %feature("docstring",
    "
    Sets the tensor info to be constant.

    Args:
        IsConstant (bool): Sets tensor info to constant.

    ") SetConstant;
    void SetConstant(const bool IsConstant = True);



    %feature("docstring",
    "
    Check that the types are the same and, if quantize, that the quantization parameters are the same.

    Returns:
        bool: True if matched, else False.

    ") IsTypeSpaceMatch;
    bool IsTypeSpaceMatch(const TensorInfo& other) const;

    %feature("docstring",
    "
    Get the number of bytes needed for this tensor.

    Returns:
        int: Number of bytes consumed by this tensor.

    ") GetNumBytes;
    unsigned int GetNumBytes() const;

};

%extend TensorInfo {

    std::string __str__() {
        const std::string tmp = "TensorInfo{DataType: " + std::to_string(static_cast<int>($self->GetDataType()))
                        + ", IsQuantized: " + std::to_string($self->IsQuantized())
                        + ", QuantizationScale: " + std::to_string( $self->GetQuantizationScale())
                        + ", QuantizationOffset: " + std::to_string($self->GetQuantizationOffset())
                        + ", IsConstant: " + std::to_string($self->IsConstant())
                        + ", NumDimensions: " + std::to_string($self->GetNumDimensions())
                        + ", NumElements: " + std::to_string($self->GetNumElements()) + "}";
        return tmp;
    }

}

class Tensor
{
public:
    ~Tensor();
    Tensor();
    Tensor(const Tensor& other);

    %mutable_memory(void* memory);
    Tensor(const TensorInfo& info, void* memory);
    %clear_mutable_memory(void* memory);

    const TensorInfo& GetInfo() const;
    const TensorShape& GetShape() const;

    DataType GetDataType() const;
    unsigned int GetNumDimensions() const;
    unsigned int GetNumBytes() const;
    unsigned int GetNumElements() const;

    /* we want to disable getting the memory area from here - forcing use of get_memory_area() in public api.
     void* GetMemoryArea() const;*/
};

%extend Tensor {

     std::string __str__() {
        const std::string tmp = "Tensor{DataType: " + std::to_string(static_cast<int>($self->GetDataType()))
                        + ", NumBytes: " + std::to_string($self->GetNumBytes())
                        + ", NumDimensions: " + std::to_string( $self->GetNumDimensions())
                        + ", NumElements: " + std::to_string($self->GetNumElements()) + "}";
        return tmp;
    }
}

class ConstTensor
{
public:
    ~ConstTensor();
    ConstTensor();
    ConstTensor(const Tensor& other);
    ConstTensor(const ConstTensor& other);

    %const_memory(const void* memory);
    ConstTensor(const TensorInfo& info, const void* memory);
    %clear_const_memory(const void* memory);

    const TensorInfo& GetInfo() const;
    const TensorShape& GetShape() const;

    DataType GetDataType() const;
    unsigned int GetNumDimensions() const;
    unsigned int GetNumBytes() const;
    unsigned int GetNumElements() const;

    /* we want to disable getting the memory area from here - forcing use of get_memory_area() in public api.
     void* GetMemoryArea() const;*/
};

%extend ConstTensor {

    std::string __str__() {
        const std::string tmp = "ConstTensor{DataType: " + std::to_string(static_cast<int>($self->GetDataType()))
                        + ", NumBytes: " + std::to_string($self->GetNumBytes())
                        + ", NumDimensions: " + std::to_string( $self->GetNumDimensions())
                        + ", NumElements: " + std::to_string($self->GetNumElements()) + "}";
        return tmp;
    }
}

}
