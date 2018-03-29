//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <array>

namespace armnn
{

constexpr unsigned int MaxNumOfTensorDimensions = 4U;

/// @enum Status enumeration
/// @var Status::Successful
/// @var Status::Failure
enum class Status
{
    Success = 0,
    Failure = 1
};

enum class DataType
{
    Float32   = 0,
    QuantisedAsymm8 = 1,
    Signed32  = 2
};

enum class ActivationFunction
{
    Sigmoid     = 0,
    TanH        = 1,
    Linear      = 2,
    ReLu        = 3,
    BoundedReLu = 4, //< min(a, max(b, input))
    SoftReLu    = 5,
    LeakyReLu   = 6,
    Abs         = 7,
    Sqrt        = 8,
    Square      = 9
};

enum class PoolingAlgorithm
{
    Max     = 0,
    Average = 1,
    L2      = 2
};

///
/// The padding method modifies the output of pooling layers.
/// In both supported methods, the values are ignored (they are
/// not even zeros which would make a difference for max pooling
/// a tensor with negative values). The difference between
/// IgnoreValue and Exclude is that the former count the padding
/// fields in the divisor of Average and L2 pooling, while
/// Exclude does not.
///
enum class PaddingMethod
{
    IgnoreValue = 0, // The padding fields count, but ignored
    Exclude     = 1  // The padding fields don't count and ignored
};

enum class NormalizationAlgorithmChannel
{
    Across = 0,
    Within = 1
};

enum class NormalizationAlgorithmMethod
{
    LocalBrightness = 0, /* Krichevsky 2012: Local Brightness Normalization */
    LocalContrast = 1  /* Jarret 2009: Local Contrast Normalization       */
};

enum class OutputShapeRounding
{
    Floor       = 0,
    Ceiling     = 1
};

enum class Compute
{
    CpuRef      = 0,  // CPU Execution: Reference C++ kernels
    CpuAcc      = 1,  // CPU Execution: NEON: ArmCompute
    GpuAcc      = 2,  // GPU Execution: OpenCL: ArmCompute
    Undefined   = 5
};

struct DeviceSpec
{
    Compute DefaultComputeDevice;
};

/// Type of identifiers for bindable layers (inputs, outputs).
using LayerBindingId = int;

class PermutationVector
{
public:
    using ValueType = unsigned int;
    using SizeType = unsigned int;
    using ArrayType = std::array<ValueType, MaxNumOfTensorDimensions>;
    using ConstIterator = typename ArrayType::const_iterator;

    /// @param dimMappings Indicates how to translate tensor elements from a given source into the target destination,
    /// when source and target potentially have different memory layouts.
    ///
    /// E.g. For a 4-d tensor laid out in memory with format (Batch Element, Height, Width, Channels),
    /// which is to be passed as an input to ArmNN, each source dimension is mapped to the corresponding
    /// ArmNN dimension. The Batch dimension remains the same (0 -> 0). The source Height dimension is mapped
    /// to the location of the ArmNN Height dimension (1 -> 2). Similar arguments are made for the Width and
    /// Channels (2 -> 3 and 3 -> 1). This will lead to @ref m_DimMappings pointing to the following array:
    /// [ 0, 2, 3, 1 ].
    ///
    /// Note that the mapping should be reversed if considering the case of ArmNN 4-d outputs (Batch Element,
    /// Channels, Height, Width) being written to a destination with the format mentioned above. We now have
    /// 0 -> 0, 2 -> 1, 3 -> 2, 1 -> 3, which, when reordered, lead to the following @ref m_DimMappings contents:
    /// [ 0, 3, 1, 2 ].
    ///
    PermutationVector(const ValueType *dimMappings, SizeType numDimMappings);

    PermutationVector(std::initializer_list<ValueType> dimMappings);

    ValueType operator[](SizeType i) const { return m_DimMappings.at(i); }

    SizeType GetSize() const { return m_NumDimMappings; }

    ConstIterator begin() const { return m_DimMappings.begin(); }
    ConstIterator end() const { return m_DimMappings.end(); }

    bool IsEqual(const PermutationVector& other) const
    {
        return std::equal(begin(), end(), other.begin(), other.end());
    }

    bool IsInverse(const PermutationVector& other) const
    {
        bool isInverse = (GetSize() == other.GetSize());
        for (SizeType i = 0; isInverse && (i < GetSize()); ++i)
        {
            isInverse = (m_DimMappings[other.m_DimMappings[i]] == i);
        }
        return isInverse;
    }

private:
    ArrayType m_DimMappings;
    /// Number of valid entries in @ref m_DimMappings
    SizeType m_NumDimMappings;
};

// Define LayerGuid type.
using LayerGuid = unsigned int;

}
