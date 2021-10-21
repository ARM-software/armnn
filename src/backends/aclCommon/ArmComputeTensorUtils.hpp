//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Tensor.hpp>
#include <armnn/DescriptorsFwd.hpp>

#include <armnn/utility/NumericCast.hpp>

#include <arm_compute/core/ITensor.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/Types.h>

#include <Half.hpp>

namespace armnn
{
class ITensorHandle;

namespace armcomputetensorutils
{

/// Utility function to map an armnn::DataType to corresponding arm_compute::DataType.
arm_compute::DataType GetArmComputeDataType(armnn::DataType dataType, bool multiScales);

/// Utility function used to set up an arm_compute::Coordinates from a vector of ArmNN Axes for reduction functions
arm_compute::Coordinates BuildArmComputeReductionCoordinates(size_t inputDimensions,
                                                             unsigned int originalInputRank,
                                                             const std::vector<unsigned int>& armnnAxes);

/// Utility function used to setup an arm_compute::TensorShape object from an armnn::TensorShape.
arm_compute::TensorShape BuildArmComputeTensorShape(const armnn::TensorShape& tensorShape);

/// Utility function used to setup an arm_compute::ITensorInfo object whose dimensions are based on the given
/// armnn::ITensorInfo.
arm_compute::TensorInfo BuildArmComputeTensorInfo(const armnn::TensorInfo& tensorInfo);

/// Utility function used to setup an arm_compute::ITensorInfo object whose dimensions are based on the given
/// armnn::ITensorInfo.
/// armnn::DataLayout.
arm_compute::TensorInfo BuildArmComputeTensorInfo(const armnn::TensorInfo& tensorInfo,
                                                  armnn::DataLayout dataLayout);

/// Utility function used to convert armnn::DataLayout to arm_compute::DataLayout
/// armnn::DataLayout.
arm_compute::DataLayout ConvertDataLayout(armnn::DataLayout dataLayout);

/// Utility function used to setup an arm_compute::PoolingLayerInfo object from given
/// armnn::Pooling2dDescriptor
/// bool fpMixedPrecision
arm_compute::PoolingLayerInfo BuildArmComputePoolingLayerInfo(const Pooling2dDescriptor& descriptor,
                                                              bool fpMixedPrecision = false);

/// Utility function to setup an arm_compute::NormalizationLayerInfo object from an armnn::NormalizationDescriptor.
arm_compute::NormalizationLayerInfo BuildArmComputeNormalizationLayerInfo(const NormalizationDescriptor& desc);

/// Utility function used to setup an arm_compute::PermutationVector object from an armnn::PermutationVector.
arm_compute::PermutationVector BuildArmComputePermutationVector(const armnn::PermutationVector& vector);

/// Utility function used to setup an arm_compute::PermutationVector object from an armnn::PermutationVector.
arm_compute::PermutationVector BuildArmComputeTransposeVector(const armnn::PermutationVector& vector);

/// Utility function used to setup an arm_compute::Size2D object from width and height values.
arm_compute::Size2D BuildArmComputeSize2D(const unsigned int width, const unsigned int height);

/// Gets the appropriate PixelValue for the TensorInfo DataType
arm_compute::PixelValue GetPixelValue(const arm_compute::ITensorInfo* tensorInfo, float pixelValue);

/// Utility function used to setup an arm_compute::PadStrideInfo object from an armnn layer descriptor.
template <typename Descriptor>
arm_compute::PadStrideInfo BuildArmComputePadStrideInfo(const Descriptor &descriptor)
{
    return arm_compute::PadStrideInfo(descriptor.m_StrideX,
                                      descriptor.m_StrideY,
                                      descriptor.m_PadLeft,
                                      descriptor.m_PadRight,
                                      descriptor.m_PadTop,
                                      descriptor.m_PadBottom,
                                      arm_compute::DimensionRoundingType::FLOOR);
}

/// Sets up the given ArmCompute tensor's dimensions based on the given ArmNN tensor.
template <typename Tensor>
void BuildArmComputeTensor(Tensor& tensor, const armnn::TensorInfo& tensorInfo)
{
    tensor.allocator()->init(BuildArmComputeTensorInfo(tensorInfo));
}

/// Sets up the given ArmCompute tensor's dimensions based on the given ArmNN tensor.
template <typename Tensor>
void BuildArmComputeTensor(Tensor& tensor, const armnn::TensorInfo& tensorInfo, DataLayout dataLayout)
{
    tensor.allocator()->init(BuildArmComputeTensorInfo(tensorInfo, dataLayout));
}

template <typename Tensor>
void InitialiseArmComputeTensorEmpty(Tensor& tensor)
{
    tensor.allocator()->allocate();
}

/// Utility function to free unused tensors after a workload is configured and prepared
template <typename Tensor>
void FreeTensorIfUnused(std::unique_ptr<Tensor>& tensor)
{
    if (tensor && !tensor->is_used())
    {
        tensor.reset(nullptr);
    }
}

// Helper function to obtain byte offset into tensor data
inline size_t GetTensorOffset(const arm_compute::ITensorInfo& info,
                              uint32_t depthIndex,
                              uint32_t batchIndex,
                              uint32_t channelIndex,
                              uint32_t y,
                              uint32_t x)
{
    arm_compute::Coordinates coords;
    coords.set(4, static_cast<int>(depthIndex));
    coords.set(3, static_cast<int>(batchIndex));
    coords.set(2, static_cast<int>(channelIndex));
    coords.set(1, static_cast<int>(y));
    coords.set(0, static_cast<int>(x));
    return armnn::numeric_cast<size_t>(info.offset_element_in_bytes(coords));
}

// Helper function to obtain element offset into data buffer representing tensor data (assuming no strides).
inline size_t GetLinearBufferOffset(const arm_compute::ITensorInfo& info,
                                    uint32_t depthIndex,
                                    uint32_t batchIndex,
                                    uint32_t channelIndex,
                                    uint32_t y,
                                    uint32_t x)
{
    const arm_compute::TensorShape& shape = info.tensor_shape();
    uint32_t width = static_cast<uint32_t>(shape[0]);
    uint32_t height = static_cast<uint32_t>(shape[1]);
    uint32_t numChannels = static_cast<uint32_t>(shape[2]);
    uint32_t numBatches = static_cast<uint32_t>(shape[3]);
    return (((depthIndex * numBatches + batchIndex) * numChannels + channelIndex) * height + y) * width + x;
}

template <typename T>
void CopyArmComputeITensorData(const arm_compute::ITensor& srcTensor, T* dstData)
{
    // If MaxNumOfTensorDimensions is increased, this loop will need fixing.
    static_assert(MaxNumOfTensorDimensions == 5, "Please update CopyArmComputeITensorData");
    {
        const arm_compute::ITensorInfo& info = *srcTensor.info();
        const arm_compute::TensorShape& shape = info.tensor_shape();
        const uint8_t* const bufferPtr = srcTensor.buffer();
        uint32_t width = static_cast<uint32_t>(shape[0]);
        uint32_t height = static_cast<uint32_t>(shape[1]);
        uint32_t numChannels = static_cast<uint32_t>(shape[2]);
        uint32_t numBatches = static_cast<uint32_t>(shape[3]);
        uint32_t depth = static_cast<uint32_t>(shape[4]);

        for (unsigned int depthIndex = 0; depthIndex < depth; ++depthIndex)
        {
            for (unsigned int batchIndex = 0; batchIndex < numBatches; ++batchIndex)
            {
                for (unsigned int channelIndex = 0; channelIndex < numChannels; ++channelIndex)
                {
                    for (unsigned int y = 0; y < height; ++y)
                    {
                        // Copies one row from arm_compute tensor buffer to linear memory buffer.
                        // A row is the largest contiguous region we can copy, as the tensor data may be using strides.
                        memcpy(
                         dstData + GetLinearBufferOffset(info, depthIndex, batchIndex, channelIndex, y, 0),
                         bufferPtr + GetTensorOffset(info, depthIndex, batchIndex, channelIndex, y, 0),
                         width * sizeof(T));
                    }
                }
            }
        }
    }
}

template <typename T>
void CopyArmComputeITensorData(const T* srcData, arm_compute::ITensor& dstTensor)
{
    // If MaxNumOfTensorDimensions is increased, this loop will need fixing.
    static_assert(MaxNumOfTensorDimensions == 5, "Please update CopyArmComputeITensorData");
    {
        const arm_compute::ITensorInfo& info = *dstTensor.info();
        const arm_compute::TensorShape& shape = info.tensor_shape();
        uint8_t* const bufferPtr = dstTensor.buffer();
        uint32_t width = static_cast<uint32_t>(shape[0]);
        uint32_t height = static_cast<uint32_t>(shape[1]);
        uint32_t numChannels = static_cast<uint32_t>(shape[2]);
        uint32_t numBatches = static_cast<uint32_t>(shape[3]);
        uint32_t depth = static_cast<uint32_t>(shape[4]);

        for (unsigned int depthIndex = 0; depthIndex < depth; ++depthIndex)
        {
            for (unsigned int batchIndex = 0; batchIndex < numBatches; ++batchIndex)
            {
                for (unsigned int channelIndex = 0; channelIndex < numChannels; ++channelIndex)
                {
                    for (unsigned int y = 0; y < height; ++y)
                    {
                        // Copies one row from linear memory buffer to arm_compute tensor buffer.
                        // A row is the largest contiguous region we can copy, as the tensor data may be using strides.
                        memcpy(
                         bufferPtr + GetTensorOffset(info, depthIndex, batchIndex, channelIndex, y, 0),
                         srcData + GetLinearBufferOffset(info, depthIndex, batchIndex, channelIndex, y, 0),
                         width * sizeof(T));
                    }
                }
            }
        }
    }
}

/// Construct a TensorShape object from an ArmCompute object based on arm_compute::Dimensions.
/// \tparam ArmComputeType Any type that implements the Dimensions interface
/// \tparam T Shape value type
/// \param shapelike An ArmCompute object that implements the Dimensions interface
/// \param initial A default value to initialise the shape with
/// \return A TensorShape object filled from the Acl shapelike object.
template<typename ArmComputeType, typename T>
TensorShape GetTensorShape(const ArmComputeType& shapelike, T initial)
{
    std::vector<unsigned int> s(MaxNumOfTensorDimensions, initial);
    for (unsigned int i=0; i < shapelike.num_dimensions(); ++i)
    {
        s[(shapelike.num_dimensions()-1)-i] = armnn::numeric_cast<unsigned int>(shapelike[i]);
    }
    return TensorShape(armnn::numeric_cast<unsigned int>(shapelike.num_dimensions()), s.data());
};

/// Get the strides from an ACL strides object
inline TensorShape GetStrides(const arm_compute::Strides& strides)
{
    return GetTensorShape(strides, 0U);
}

/// Get the shape from an ACL shape object
inline TensorShape GetShape(const arm_compute::TensorShape& shape)
{
    return GetTensorShape(shape, 1U);
}

} // namespace armcomputetensorutils
} // namespace armnn
