//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "TensorFwd.hpp"

#include "Exceptions.hpp"
#include "Types.hpp"

#include <array>
#include <initializer_list>
#include <vector>

namespace armnn
{

class TensorShape
{
public:
    /// Empty (invalid) constructor.
    TensorShape();

    TensorShape(unsigned int numDimensions);

    TensorShape(unsigned int numDimensions, const unsigned int* dimensionSizes);

    TensorShape(std::initializer_list<unsigned int> dimensionSizeList);

    TensorShape(const TensorShape& other);

    TensorShape& operator=(const TensorShape& other);

    unsigned int operator[](unsigned int i) const;

    unsigned int& operator[](unsigned int i);

    bool operator==(const TensorShape& other) const;
    bool operator!=(const TensorShape& other) const;

    unsigned int GetNumDimensions() const { return m_NumDimensions; }
    unsigned int GetNumElements() const;

private:
    std::array<unsigned int, MaxNumOfTensorDimensions> m_Dimensions;
    unsigned int m_NumDimensions;

    void CheckDimensionIndex(unsigned int i) const;
};

class TensorInfo
{
public:
    /// Empty (invalid) constructor.
    TensorInfo();

    TensorInfo(const TensorShape& shape, DataType dataType,
        float quantizationScale = 0.0f, int32_t quantizationOffset = 0);
    TensorInfo(unsigned int numDimensions, const unsigned int* dimensionSizes, DataType dataType,
        float quantizationScale = 0.0f, int32_t quantizationOffset = 0);

    TensorInfo(const TensorInfo& other);

    TensorInfo& operator=(const TensorInfo& other);

    bool operator==(const TensorInfo& other) const;
    bool operator!=(const TensorInfo& other) const;

    const TensorShape& GetShape() const             { return m_Shape; }
    TensorShape& GetShape()                         { return m_Shape; }
    void SetShape(const TensorShape& newShape)      { m_Shape = newShape; }

    unsigned int GetNumDimensions() const { return m_Shape.GetNumDimensions(); }
    unsigned int GetNumElements() const { return m_Shape.GetNumElements(); }

    DataType GetDataType() const                    { return m_DataType; }
    void SetDataType(DataType type)                 { m_DataType = type; }

    float GetQuantizationScale() const              { return m_Quantization.m_Scale; }
    int32_t GetQuantizationOffset() const           { return m_Quantization.m_Offset; }
    void SetQuantizationScale(float scale)          { m_Quantization.m_Scale = scale; }
    void SetQuantizationOffset(int32_t offset)      { m_Quantization.m_Offset = offset; }
    bool IsQuantized() const                        { return m_DataType == DataType::QuantisedAsymm8 ||
                                                             m_DataType == DataType::QuantisedSymm16; }

    /// Check that the types are the same and, if quantize, that the quantization parameters are the same.
    bool IsTypeSpaceMatch(const TensorInfo& other) const;

    unsigned int GetNumBytes() const;

private:
    TensorShape m_Shape;
    DataType m_DataType;
    /// Scale and offset values are used for quantization.
    struct Quantization
    {
        Quantization() : m_Scale(0.f), m_Offset(0) {}
        bool operator==(const Quantization& o) const {return ((m_Scale == o.m_Scale) && (m_Offset == o.m_Offset));}
        float m_Scale;
        int32_t m_Offset;
    } m_Quantization;
};

using BindingPointInfo = std::pair<armnn::LayerBindingId, armnn::TensorInfo>;

template<typename MemoryType>
class BaseTensor
{
public:
    /// Empty (invalid) constructor.
    BaseTensor();

    /// Constructor from a raw memory pointer.
    /// @param memoryArea - Region of CPU-addressable memory where tensor data will be stored. Must be valid while
    /// workloads are on the fly. Tensor instances do not claim ownership of referenced memory regions, that is,
    /// no attempt will be made by ArmNN to free these memory regions automatically.
    BaseTensor(const TensorInfo& info, MemoryType memoryArea);

    /// Tensors are copyable.
    BaseTensor(const BaseTensor& other);

    /// Tensors are copyable.
    BaseTensor& operator=(const BaseTensor&);

    const TensorInfo& GetInfo() const { return m_Info; }
    TensorInfo& GetInfo() { return m_Info; }
    const TensorShape& GetShape() const { return m_Info.GetShape(); }
    TensorShape& GetShape() { return m_Info.GetShape(); }

    DataType GetDataType() const                    { return m_Info.GetDataType(); }
    unsigned int GetNumDimensions() const { return m_Info.GetNumDimensions(); }
    unsigned int GetNumBytes() const { return m_Info.GetNumBytes(); }
    unsigned int GetNumElements() const { return m_Info.GetNumElements(); }

    MemoryType GetMemoryArea() const { return m_MemoryArea; }

protected:
    // Protected destructor to stop users from making these
    // (could still new one on the heap and then leak it...)
    ~BaseTensor() {}

    MemoryType m_MemoryArea;

private:
    TensorInfo m_Info;
};

/// A tensor defined by a TensorInfo (shape and data type) and a mutable backing store.
class Tensor : public BaseTensor<void*>
{
public:
    /// Brings in the constructors and assignment operator.
    using BaseTensor<void*>::BaseTensor; 
};

/// A tensor defined by a TensorInfo (shape and data type) and an immutable backing store.
class ConstTensor : public BaseTensor<const void*>
{
public:
    /// Brings in the constructors and assignment operator.
    using BaseTensor<const void*>::BaseTensor; 
    ConstTensor() : BaseTensor<const void*>() {} // This needs to be redefined explicitly??

    /// Can be implicitly constructed from non-const Tensor.
    ConstTensor(const Tensor& other) : BaseTensor<const void*>(other.GetInfo(), other.GetMemoryArea()) {}

    /// Constructor from a backing container.
    /// @param container - An stl-like container type which implements data() and size() methods.
    /// Presence of data() and size() is a strong indicator of the continuous memory layout of the container,
    /// which is a requirement for Tensor data. Tensor instances do not claim ownership of referenced memory regions,
    /// that is, no attempt will be made by ArmNN to free these memory regions automatically.
    template < template<typename, typename...> class ContainerType, typename T, typename...ContainerArgs >
    ConstTensor(const TensorInfo& info, const ContainerType<T, ContainerArgs...>& container)
        : BaseTensor<const void*>(info, container.data())
    {
        if (container.size() * sizeof(T) != info.GetNumBytes())
        {
            throw InvalidArgumentException("Container size is not correct");
        }
    }
};

using InputTensors = std::vector<std::pair<LayerBindingId, class ConstTensor>>;
using OutputTensors = std::vector<std::pair<LayerBindingId, class Tensor>>;

} // namespace armnn
