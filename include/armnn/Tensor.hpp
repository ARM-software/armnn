//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "TensorFwd.hpp"

#include "Exceptions.hpp"
#include "Optional.hpp"
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

    /// Constructor for TensorShape
    /// @param numDimensions - Tensor rank.
    /// @param initDimensionsSpecificity (optional) - value to initialize the specificity of each dimension size.
    explicit TensorShape(unsigned int numDimensions, bool initDimensionsSpecificity = true);

    /// Constructor for TensorShape
    /// @param numDimensions - Tensor rank.
    /// @param dimensionSizes - Size of each of dimension.
    TensorShape(unsigned int numDimensions, const unsigned int* dimensionSizes);

    /// Constructor for TensorShape
    /// @param dimensionSizeList - Size of each of dimension.
    TensorShape(std::initializer_list<unsigned int> dimensionSizeList);

    /// Copy Constructor for TensorShape
    /// @param other - TensorShape to copy from.
    TensorShape(const TensorShape& other);

    /// Constructor for TensorShape
    /// @param numDimensions - Tensor rank.
    /// @param dimensionSizes - Size of each of dimension.
    /// @param dimensionsSpecificity - Flags to indicate which dimension has its size specified.
    TensorShape(unsigned int numDimensions, const unsigned int* dimensionSizes, const bool* dimensionsSpecificity);

    /// Constructor for TensorShape
    /// @param dimensionSizeList - Size of each of dimension.
    /// @param dimensionsSpecificityList - Flags to indicate which dimension size is specified.
    TensorShape(std::initializer_list<unsigned int> dimensionSizeList,
                std::initializer_list<bool> dimensionsSpecificityList);

    /// Constructor for TensorShape
    /// @param dimensionality - Parameter to indicate if the Tensor is a Scalar, a Tensor of known dimensionality
    /// or a Tensor of unknown dimensionality.
    explicit TensorShape(Dimensionality dimensionality);

    /// Assignation function
    /// @param other - TensorShape to copy from.
    TensorShape& operator=(const TensorShape& other);

    /// Read only operator
    /// @param i - Dimension index.
    unsigned int operator[](unsigned int i) const;

    /// Read and write operator
    /// @param i - Dimension index.
    unsigned int& operator[](unsigned int i);

    /// Equality comparison operator
    /// @param other - TensorShape to compare with.
    bool operator==(const TensorShape& other) const;

    /// Inequality comparison operator
    /// @param other - TensorShape to compare with.
    bool operator!=(const TensorShape& other) const;

    /// Function that returns the tensor rank.
    /// @return - Tensor rank.
    unsigned int GetNumDimensions() const;

    /// Function that calculates the tensor elements by multiplying all dimension size which are Specified.
    /// @return - Total number of elements in the tensor.
    unsigned int GetNumElements() const;

    /// Function that returns the tensor type.
    /// @return - Parameter to indicate if the Tensor is a scalar, a Tensor of known dimensionality or
    /// a Tensor of unknown dimensionality
    Dimensionality GetDimensionality() const { return m_Dimensionality; }

    /// Gets information about if the dimension size has been specified or not
    /// @param i - Dimension index.
    /// @return - Flag to indicate if the dimension "i" has a specified size.
    bool GetDimensionSpecificity(unsigned int i) const;

    /// Sets the tensor rank and therefore the Dimensionality is set to Specified if it was not.
    /// @param numDimensions - Tensor rank.
    /// @param initDimensionsSpecificity (optional) - value to initialize the specificity of each dimension size.
    void SetNumDimensions(unsigned int numDimensions, bool initDimensionsSpecificity = false);

    /// Sets the size of the indicated dimension and Specificity for that dimension is set to true.
    /// @param i - Dimension index.
    /// @param dimensionSize - size of one dimension.
    void SetDimensionSize(unsigned int i, unsigned int dimensionSize);

    /// Checks if there is at least one dimension not specified. AND of all array elements.
    /// @return - True when all dimension sizes are specified. False when at least one dimension size is not specified.
    bool AreAllDimensionsSpecified() const;

    /// Checks if there is at least one dimension specified. OR of all array elements.
    /// @return - True at least one dimension sizes is specified. False when all dimension sizes are not specified.
    bool IsAtLeastOneDimensionSpecified() const;

private:
    /// Array of the dimension sizes.
    std::array<unsigned int, MaxNumOfTensorDimensions> m_Dimensions{};

    /// Array of flags to indicate if the size of each of the dimensions is specified or not
    std::array<bool, MaxNumOfTensorDimensions> m_DimensionsSpecificity = { {true} };

    /// Tensor rank
    unsigned int m_NumDimensions{};

    /// Tensor type: Specified, NotSpecified or Scalar.
    Dimensionality m_Dimensionality = Dimensionality::Specified;

    /// Checks if the dimension index given is within range.
    /// @param i - Dimension index.
    void CheckDimensionIndex(unsigned int i) const;

    /// Checks if the tensor rank given is within range.
    /// @param numDimensions - Tensor rank.
    static void CheckValidNumDimensions(unsigned int numDimensions) ;

    /// Checks if the size of the dimension index given is specified.
    /// @param i - Dimension index.
    void CheckDimensionSpecified(unsigned int i) const;

    /// Checks if this is a scalar.
    void CheckScalar() const;

    /// Checks if the number of dimensions is unknown, i.e. rank is unspecified.
    void CheckUnspecifiedNumDimensions() const;

    /// Checks if the number of dimensions is known, i.e. rank is specified.
    void CheckSpecifiedNumDimensions() const;
};

class TensorInfo
{
public:
    /// Empty (invalid) constructor.
    TensorInfo();

    TensorInfo(const TensorShape& shape,
               DataType dataType,
               float quantizationScale = 0.0f,
               int32_t quantizationOffset = 0);

    TensorInfo(unsigned int numDimensions,
               const unsigned int* dimensionSizes,
               DataType dataType,
               float quantizationScale = 0.0f,
               int32_t quantizationOffset = 0);

    TensorInfo(const TensorShape& shape,
               DataType dataType,
               const std::vector<float>& quantizationScales,
               unsigned int quantizationDim);

    TensorInfo(unsigned int numDimensions,
               const unsigned int* dimensionSizes,
               DataType dataType,
               const std::vector<float>& quantizationScales,
               unsigned int quantizationDim);

    TensorInfo(const TensorInfo& other);

    TensorInfo& operator=(const TensorInfo& other);

    bool operator==(const TensorInfo& other) const;
    bool operator!=(const TensorInfo& other) const;

    const TensorShape& GetShape() const              { return m_Shape; }
    TensorShape& GetShape()                          { return m_Shape; }
    void SetShape(const TensorShape& newShape)       { m_Shape = newShape; }

    unsigned int GetNumDimensions() const            { return m_Shape.GetNumDimensions(); }
    unsigned int GetNumElements() const              { return m_Shape.GetNumElements(); }

    DataType GetDataType() const                     { return m_DataType; }
    void SetDataType(DataType type)                  { m_DataType = type; }

    bool HasMultipleQuantizationScales() const       { return m_Quantization.m_Scales.size() > 1; }

    bool HasPerAxisQuantization() const;

    std::vector<float> GetQuantizationScales() const;
    void SetQuantizationScales(const std::vector<float>& scales);

    float GetQuantizationScale() const;
    void SetQuantizationScale(float scale);

    int32_t GetQuantizationOffset() const;
    void SetQuantizationOffset(int32_t offset);

    Optional<unsigned int> GetQuantizationDim() const;
    void SetQuantizationDim(const Optional<unsigned int>& quantizationDim);

    bool IsQuantized() const;

    /// Check that the types are the same and, if quantize, that the quantization parameters are the same.
    bool IsTypeSpaceMatch(const TensorInfo& other) const;

    unsigned int GetNumBytes() const;

private:
    TensorShape m_Shape;
    DataType    m_DataType;

    /// Vectors of scale and offset are used for per-axis quantization.
    struct Quantization
    {
        Quantization()
            : m_Scales{}
            , m_Offset(EmptyOptional())
            , m_QuantizationDim(EmptyOptional()) {}

        Quantization(const Quantization& other)
            : m_Scales(other.m_Scales)
            , m_Offset(other.m_Offset)
            , m_QuantizationDim(other.m_QuantizationDim) {}

        bool operator==(const Quantization& other) const
        {
            return ((m_Scales == other.m_Scales) && (m_Offset == other.m_Offset) &&
                (m_QuantizationDim == other.m_QuantizationDim));
        }

        Quantization& operator=(const Quantization& other)
        {
            if(this != &other)
            {
                m_Scales = other.m_Scales;
                m_Offset = other.m_Offset;
                m_QuantizationDim = other.m_QuantizationDim;
            }
            return *this;
        }

        std::vector<float>     m_Scales;
        Optional<int32_t>      m_Offset;
        Optional<unsigned int> m_QuantizationDim;

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
    /// Protected destructor to stop users from making these
    /// (could still new one on the heap and then leak it...)
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
