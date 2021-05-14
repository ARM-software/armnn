//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnn/Tensor.hpp"
#include "armnn/Utils.hpp"
#include "armnn/Exceptions.hpp"
#include "armnn/TypesUtils.hpp"

#include <armnn/utility/Assert.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <iostream>

#include <sstream>

namespace armnn
{

// ---
// --- TensorShape
// ---

TensorShape::TensorShape()
 : m_NumDimensions(0), m_Dimensionality(Dimensionality::Specified)
{
}

TensorShape::TensorShape(unsigned int numDimensions, bool initDimensionsSpecificity)
 : m_NumDimensions(numDimensions), m_Dimensionality(Dimensionality::Specified)
{
    CheckValidNumDimensions(numDimensions);

    std::fill(m_Dimensions.begin(), m_Dimensions.begin() + m_NumDimensions, 0);
    std::fill(m_DimensionsSpecificity.begin(), m_DimensionsSpecificity.begin() + m_NumDimensions,
              initDimensionsSpecificity);
}

TensorShape::TensorShape(const unsigned int numDimensions, const unsigned int* const dimensionSizes)
 : m_NumDimensions(numDimensions), m_Dimensionality(Dimensionality::Specified)
{
    CheckValidNumDimensions(numDimensions);

    if (dimensionSizes == nullptr)
    {
        throw InvalidArgumentException("Tensor dimensionSizes must not be NULL");
    }

    std::copy(dimensionSizes, dimensionSizes + numDimensions, m_Dimensions.begin());
    std::fill(m_DimensionsSpecificity.begin(), m_DimensionsSpecificity.begin() + m_NumDimensions, true);
}

TensorShape::TensorShape(std::initializer_list<unsigned int> dimensionSizeList)
 : TensorShape(armnn::numeric_cast<unsigned int>(dimensionSizeList.size()), dimensionSizeList.begin())
{
}

TensorShape::TensorShape(unsigned int numDimensions,
                         const unsigned int* const dimensionSizes,
                         const bool* const dimensionsSpecificity)
                       : m_NumDimensions(numDimensions), m_Dimensionality(Dimensionality::Specified)
{
    CheckValidNumDimensions(numDimensions);

    if (dimensionSizes == nullptr)
    {
        throw InvalidArgumentException("Tensor dimensionSizes must not be NULL");
    }

    if (dimensionsSpecificity == nullptr)
    {
        throw InvalidArgumentException("Tensor dimensionsSpecificity must not be NULL");
    }

    std::copy(dimensionSizes, dimensionSizes + numDimensions, m_Dimensions.begin());
    std::copy(dimensionsSpecificity, dimensionsSpecificity + numDimensions, m_DimensionsSpecificity.begin());
}

TensorShape::TensorShape(std::initializer_list<unsigned int> dimensionSizeList,
                         std::initializer_list<bool> dimensionsSpecificityList)
{
    auto numDimensions = static_cast<unsigned int>(dimensionSizeList.size());
    if (dimensionsSpecificityList.size() != numDimensions)
    {
        throw InvalidArgumentException("Tensors dimensionSizeList and dimensionsSpecificityList must be same size");
    }

    *this = TensorShape(numDimensions, dimensionSizeList.begin(), dimensionsSpecificityList.begin());
}

TensorShape::TensorShape(Dimensionality dimensionality)
: m_Dimensionality(dimensionality)
{
    switch (dimensionality)
    {
        case Dimensionality::Specified:
            throw InvalidArgumentException("Use other constructor to specify the rest of the values, this one is only "
                                           "for tensors that have an unknown number of dimensions or that are scalar");
            break;
        case Dimensionality::NotSpecified:
            m_NumDimensions = 0;
            m_Dimensions = {0};
            m_DimensionsSpecificity = {false};
            break;
        case Dimensionality::Scalar:
            m_NumDimensions = 1;
            m_Dimensions = {1};
            m_DimensionsSpecificity = {true};
            break;
        default:
            throw InvalidArgumentException("Invalid Dimensionality value");
    }
}

TensorShape::TensorShape(const TensorShape& other)
 : m_NumDimensions(other.m_NumDimensions), m_Dimensionality(other.m_Dimensionality)
{
    std::copy(other.m_Dimensions.cbegin(), other.m_Dimensions.cbegin() + other.m_NumDimensions, m_Dimensions.begin());
    std::copy(other.m_DimensionsSpecificity.cbegin(), other.m_DimensionsSpecificity.cbegin() + other.m_NumDimensions,
              m_DimensionsSpecificity.begin());
}

TensorShape& TensorShape::operator =(const TensorShape& other)
{
    m_NumDimensions = other.m_NumDimensions;
    m_Dimensionality = other.m_Dimensionality;
    std::copy(other.m_Dimensions.cbegin(), other.m_Dimensions.cbegin() + other.m_NumDimensions, m_Dimensions.begin());
    std::copy(other.m_DimensionsSpecificity.cbegin(), other.m_DimensionsSpecificity.cbegin() + other.m_NumDimensions,
              m_DimensionsSpecificity.begin());
    return *this;
}

// read
unsigned int TensorShape::operator[](unsigned int i) const
{
    CheckUnspecifiedNumDimensions();
    CheckDimensionIndex(i);
    CheckDimensionSpecified(i);

    return m_Dimensions.at(i);
}

// read and write
unsigned int& TensorShape::operator[](unsigned int i)
{
    if (Dimensionality::Scalar == m_Dimensionality)
    {
        std::stringstream errorMessage;
        errorMessage << "TensorShape with Dimensionality::Scalar must be const to use operator[]";
        throw InvalidArgumentException(errorMessage.str(), CHECK_LOCATION());
    }
    CheckUnspecifiedNumDimensions();
    CheckDimensionIndex(i);
    CheckDimensionSpecified(i);

    return m_Dimensions.at(i);
}

bool TensorShape::operator==(const TensorShape& other) const
{
    return ((m_NumDimensions == other.m_NumDimensions) &&
            (m_Dimensionality == other.m_Dimensionality) &&
             std::equal(m_Dimensions.cbegin(), m_Dimensions.cbegin() + m_NumDimensions, other.m_Dimensions.cbegin()) &&
             std::equal(m_DimensionsSpecificity.cbegin(), m_DimensionsSpecificity.cbegin() + m_NumDimensions,
                        other.m_DimensionsSpecificity.cbegin()));
}

bool TensorShape::operator!=(const TensorShape& other) const
{
    return !(*this == other);
}

unsigned int TensorShape::GetNumDimensions() const
{
    CheckUnspecifiedNumDimensions();

    return m_NumDimensions;
}

unsigned int TensorShape::GetNumElements() const
{
    CheckUnspecifiedNumDimensions();

    if (m_NumDimensions == 0)
    {
        return 0;
    }

    unsigned int count = 1;
    bool atLeastOneDimensionSpecified = false;
    for (unsigned int i = 0; i < m_NumDimensions; ++i)
    {
        if (m_DimensionsSpecificity[i])
        {
            atLeastOneDimensionSpecified = true;
            count *= m_Dimensions[i];
        }
    }

    if (atLeastOneDimensionSpecified)
    {
        return count;
    } 
    else 
    {
        return 0;
    }
}

bool TensorShape:: GetDimensionSpecificity(unsigned int i) const
{
    CheckUnspecifiedNumDimensions();
    CheckDimensionIndex(i);

    return m_DimensionsSpecificity[i];
}

void TensorShape::SetNumDimensions(unsigned int numDimensions, bool initDimensionsSpecificity)
{
    CheckScalar();
    CheckSpecifiedNumDimensions();
    CheckValidNumDimensions(numDimensions);

    m_NumDimensions = numDimensions;
    m_Dimensionality = Dimensionality::Specified;
    std::fill(m_Dimensions.begin(), m_Dimensions.begin() + m_NumDimensions, 0);
    std::fill(m_DimensionsSpecificity.begin(), m_DimensionsSpecificity.begin() + m_NumDimensions,
              initDimensionsSpecificity);
}

void TensorShape::SetDimensionSize(unsigned int i, unsigned int dimensionSize)
{
    CheckScalar();
    CheckDimensionIndex(i);

    m_Dimensions[i] = dimensionSize;
    m_DimensionsSpecificity[i] = true;
}

bool TensorShape::AreAllDimensionsSpecified() const
{
    CheckUnspecifiedNumDimensions();

    bool areAllDimensionsSpecified = true;
    for (unsigned int i = 0; i < m_NumDimensions; ++i)
    {
        if (!m_DimensionsSpecificity[i])
        {
            areAllDimensionsSpecified = false;
            break;
        }
    }
    return areAllDimensionsSpecified;
}

bool TensorShape::IsAtLeastOneDimensionSpecified() const
{
    CheckUnspecifiedNumDimensions();

    bool isAtLeastOneDimensionSpecified = false;
    for (unsigned int i = 0; i < m_NumDimensions; ++i)
    {
        if (m_DimensionsSpecificity[i])
        {
            isAtLeastOneDimensionSpecified = true;
            break;
        }
    }
    return isAtLeastOneDimensionSpecified;
}

void TensorShape::CheckDimensionIndex(unsigned int i) const
{
    if (i >= m_NumDimensions)
    {
        std::stringstream errorMessage;
        errorMessage << "Invalid dimension index: " << i << " (number of dimensions is " << m_NumDimensions << ")";
        throw InvalidArgumentException(errorMessage.str(), CHECK_LOCATION());
    }
}

void TensorShape::CheckValidNumDimensions(unsigned int numDimensions)
{
    if (numDimensions < 1)
    {
        throw InvalidArgumentException("Tensor numDimensions must be greater than 0", CHECK_LOCATION());
    }

    if (numDimensions > MaxNumOfTensorDimensions)
    {
        throw InvalidArgumentException("Tensor numDimensions must be less than or equal to MaxNumOfTensorDimensions"
                , CHECK_LOCATION());
    }
}

void TensorShape::CheckDimensionSpecified(unsigned int i) const
{
    if (!m_DimensionsSpecificity[i])
    {
        std::stringstream errorMessage;
        errorMessage << "Dimension index: " << i << " not specified. Tensor shape not inferred yet.";
        throw InvalidArgumentException(errorMessage.str(), CHECK_LOCATION());
    }
}

void TensorShape::CheckScalar() const
{
    if (Dimensionality::Scalar == m_Dimensionality)
    {
        std::stringstream errorMessage;
        errorMessage << "Invalid action on a tensor shape that holds a scalar value.";
        throw InvalidArgumentException(errorMessage.str(), CHECK_LOCATION());
    }
}

void TensorShape::CheckUnspecifiedNumDimensions() const
{
    if (Dimensionality::NotSpecified == m_Dimensionality)
    {
        std::stringstream errorMessage;
        errorMessage << "Invalid action on a tensor shape that has unknown number of dimensions.";
        throw InvalidArgumentException(errorMessage.str(), CHECK_LOCATION());
    }
}

void TensorShape::CheckSpecifiedNumDimensions() const
{
    if (Dimensionality::Specified == m_Dimensionality)
    {
        std::stringstream errorMessage;
        errorMessage << "Invalid action on a tensor shape that has known number of dimensions.";
        throw InvalidArgumentException(errorMessage.str(), CHECK_LOCATION());
    }
}

// ---
// --- TensorInfo
// ---

TensorInfo::TensorInfo()
: m_DataType(DataType::Float32), m_IsConstant(false)
{
}

TensorInfo::TensorInfo(const TensorShape& shape,
                       DataType dataType,
                       float quantizationScale,
                       int32_t quantizationOffset,
                       bool isConstant)
    : m_Shape(shape)
    , m_DataType(dataType)
    , m_IsConstant(isConstant)
{
    SetQuantizationScale(quantizationScale);
    SetQuantizationOffset(quantizationOffset);
}

TensorInfo::TensorInfo(unsigned int numDimensions,
                       const unsigned int* dimensionSizes,
                       DataType dataType,
                       float quantizationScale,
                       int32_t quantizationOffset,
                       bool isConstant)
    : m_Shape(numDimensions, dimensionSizes)
    , m_DataType(dataType)
        , m_IsConstant(isConstant)
{
    SetQuantizationScale(quantizationScale);
    SetQuantizationOffset(quantizationOffset);
}

TensorInfo::TensorInfo(const TensorShape& shape,
                       DataType dataType,
                       const std::vector<float>& quantizationScales,
                       unsigned int quantizationDim,
                       bool isConstant)
    : m_Shape(shape)
    , m_DataType(dataType)
    , m_IsConstant(isConstant)
{
    SetQuantizationScales(quantizationScales);
    SetQuantizationDim(MakeOptional<unsigned int>(quantizationDim));
}

TensorInfo::TensorInfo(unsigned int numDimensions,
                       const unsigned int* dimensionSizes,
                       DataType dataType,
                       const std::vector<float>& quantizationScales,
                       unsigned int quantizationDim,
                       bool isConstant)
    : m_Shape(numDimensions, dimensionSizes)
    , m_DataType(dataType)
    , m_IsConstant(isConstant)
{
    SetQuantizationScales(quantizationScales);
    SetQuantizationDim(MakeOptional<unsigned int>(quantizationDim));
}

TensorInfo::TensorInfo(const TensorInfo& other)
: m_Shape(other.m_Shape)
, m_DataType(other.m_DataType)
, m_IsConstant(other.m_IsConstant)
, m_Quantization(other.m_Quantization)
{}

TensorInfo& TensorInfo::operator=(const TensorInfo& other)
{
    m_Shape = other.m_Shape;
    m_DataType = other.m_DataType;
    m_Quantization = other.m_Quantization;
    m_IsConstant = other.m_IsConstant;
    return *this;
}

bool TensorInfo::operator==(const TensorInfo& other) const
{
    return ((m_Shape == other.m_Shape) &&
            (m_DataType == other.m_DataType) &&
            (m_Quantization == other.m_Quantization) &&
            (m_IsConstant == other.m_IsConstant));
}

bool TensorInfo::operator!=(const TensorInfo& other) const
{
    return !(*this == other);
}

unsigned int TensorInfo::GetNumBytes() const
{
    return GetDataTypeSize(m_DataType) * GetNumElements();
}

bool TensorInfo::IsTypeSpaceMatch(const TensorInfo& other) const
{
    bool match = true;

    match &= m_DataType == other.m_DataType;

    if (IsQuantized() && !HasMultipleQuantizationScales())
    {
        match &= GetQuantizationScale() == other.GetQuantizationScale() &&
                 GetQuantizationOffset() == other.GetQuantizationOffset();
    }
    return match;
}

bool TensorInfo::HasPerAxisQuantization() const
{
    return HasMultipleQuantizationScales() || m_Quantization.m_QuantizationDim.has_value();
}

std::vector<float> TensorInfo::GetQuantizationScales() const
{
    return m_Quantization.m_Scales;
}

void TensorInfo::SetQuantizationScales(const std::vector<float>& scales)
{
    m_Quantization.m_Scales = scales;
}

float TensorInfo::GetQuantizationScale() const
{
    if (m_Quantization.m_Scales.empty())
    {
        // NOTE: old default for backward compatibility
        return 1.0f;
    }

    ARMNN_ASSERT(!HasMultipleQuantizationScales());
    return m_Quantization.m_Scales[0];
}

void TensorInfo::SetQuantizationScale(float scale)
{
    m_Quantization.m_Scales = { scale };
}

int32_t TensorInfo::GetQuantizationOffset() const
{
    if (!m_Quantization.m_Offset.has_value())
    {
        // NOTE: old default for backward compatibility
        return 0;
    }

    return m_Quantization.m_Offset.value();
}

void TensorInfo::SetQuantizationOffset(int32_t offset)
{
    m_Quantization.m_Offset = MakeOptional<int32_t>(offset);
}

Optional<unsigned int> TensorInfo::GetQuantizationDim() const
{
    return m_Quantization.m_QuantizationDim;
}

void TensorInfo::SetQuantizationDim(const Optional<unsigned int>& quantizationDim)
{
    m_Quantization.m_QuantizationDim = quantizationDim;
}

bool TensorInfo::IsQuantized() const
{
    return IsQuantizedType(m_DataType);
}

bool TensorInfo::IsConstant() const
{
    return m_IsConstant;
}

void TensorInfo::SetConstant(const bool IsConstant)
{
    m_IsConstant = IsConstant;
}

// ---
// --- BaseTensor
// ---

template<typename MemoryType>
BaseTensor<MemoryType>::BaseTensor()
 : m_MemoryArea(nullptr)
{
}

template<typename MemoryType>
BaseTensor<MemoryType>::BaseTensor(const TensorInfo& info, MemoryType memoryArea)
 : m_MemoryArea(memoryArea)
 , m_Info(info)
{
}

template<typename MemoryType>
BaseTensor<MemoryType>::BaseTensor(const BaseTensor<MemoryType>& other)
 : m_MemoryArea(other.m_MemoryArea)
 , m_Info(other.GetInfo())
{
}

template<typename MemoryType>
BaseTensor<MemoryType>& BaseTensor<MemoryType>::operator =(const BaseTensor<MemoryType>& other)
{
    m_Info = other.m_Info;
    m_MemoryArea = other.m_MemoryArea;
    return *this;
}

// Explicit instantiations.
template class BaseTensor<const void*>;
template class BaseTensor<void*>;

} // namespace armnn
