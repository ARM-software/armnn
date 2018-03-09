//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <armnn/Tensor.hpp>
#include <armnn/DescriptorsFwd.hpp>

#include <arm_compute/core/ITensor.h>
#include <arm_compute/core/TensorInfo.h>

#include <boost/cast.hpp>

namespace armnn
{
class ITensorHandle;

namespace armcomputetensorutils
{

/// Utility function to map an armnn::DataType to corresponding arm_compute::DataType
arm_compute::DataType GetArmComputeDataType(armnn::DataType dataType);

/// Utility function used to setup an arm_compute::TensorShape object from an armnn::TensorShape
arm_compute::TensorShape BuildArmComputeTensorShape(const armnn::TensorShape& tensorShape);

/// Utility function used to setup an arm_compute::ITensorInfo object whose dimensions are based on the given
/// armnn::ITensorInfo
arm_compute::TensorInfo BuildArmComputeTensorInfo(const armnn::TensorInfo& tensorInfo);

/// Utility function used to setup an arm_compute::PoolingLayerInfo object from an armnn::Pooling2dDescriptor
arm_compute::PoolingLayerInfo BuildArmComputePoolingLayerInfo(const Pooling2dDescriptor& descriptor);

/// Utility function to setup an arm_compute::NormalizationLayerInfo object from an armnn::NormalizationDescriptor
arm_compute::NormalizationLayerInfo BuildArmComputeNormalizationLayerInfo(const NormalizationDescriptor& desc);

/// Utility function used to setup an arm_compute::PermutationVector object from an armnn::PermutationVector
arm_compute::PermutationVector BuildArmComputePermutationVector(const armnn::PermutationVector& vector);

/// Sets up the given ArmCompute tensor's dimensions based on the given ArmNN tensor.
template <typename Tensor>
void BuildArmComputeTensor(Tensor& tensor, const armnn::TensorInfo& tensorInfo)
{
    tensor.allocator()->init(BuildArmComputeTensorInfo(tensorInfo));
}

template <typename Tensor>
void InitialiseArmComputeTensorEmpty(Tensor& tensor)
{
    tensor.allocator()->allocate();
}

// Helper function to obtain byte offset into tensor data
inline size_t GetTensorOffset(const arm_compute::ITensorInfo& info,
                              uint32_t batchIndex,
                              uint32_t channelIndex,
                              uint32_t y,
                              uint32_t x)
{
    arm_compute::Coordinates coords;
    coords.set(3, boost::numeric_cast<int>(batchIndex));
    coords.set(2, boost::numeric_cast<int>(channelIndex));
    coords.set(1, boost::numeric_cast<int>(y));
    coords.set(0, boost::numeric_cast<int>(x));
    return info.offset_element_in_bytes(coords);
}

// Helper function to obtain element offset into data buffer representing tensor data (assuming no strides)
inline size_t GetLinearBufferOffset(const arm_compute::ITensorInfo& info,
                                    uint32_t batchIndex,
                                    uint32_t channelIndex,
                                    uint32_t y,
                                    uint32_t x)
{
    const arm_compute::TensorShape& shape = info.tensor_shape();
    uint32_t width = boost::numeric_cast<uint32_t>(shape[0]);
    uint32_t height = boost::numeric_cast<uint32_t>(shape[1]);
    uint32_t numChannels = boost::numeric_cast<uint32_t>(shape[2]);
    return ((batchIndex * numChannels + channelIndex) * height + y) * width + x;
}

template <typename T>
void CopyArmComputeITensorData(const arm_compute::ITensor& srcTensor, T* dstData)
{
    // if MaxNumOfTensorDimensions is increased, this loop will need fixing
    static_assert(MaxNumOfTensorDimensions == 4, "Please update CopyArmComputeITensorData");
    {
        const arm_compute::ITensorInfo& info = *srcTensor.info();
        const arm_compute::TensorShape& shape = info.tensor_shape();
        const uint8_t* const bufferPtr = srcTensor.buffer();
        uint32_t width = boost::numeric_cast<uint32_t>(shape[0]);
        uint32_t height = boost::numeric_cast<uint32_t>(shape[1]);
        uint32_t numChannels = boost::numeric_cast<uint32_t>(shape[2]);
        uint32_t numBatches = boost::numeric_cast<uint32_t>(shape[3]);

        for (unsigned int batchIndex = 0; batchIndex < numBatches; ++batchIndex)
        {
            for (unsigned int channelIndex = 0; channelIndex < numChannels; ++channelIndex)
            {
                for (unsigned int y = 0; y < height; ++y)
                {
                    // Copy one row from arm_compute tensor buffer to linear memory buffer
                    // A row is the largest contiguous region we can copy, as the tensor data may be using strides
                    memcpy(dstData + GetLinearBufferOffset(info, batchIndex, channelIndex, y, 0),
                           bufferPtr + GetTensorOffset(info, batchIndex, channelIndex, y, 0),
                           width * sizeof(T));
                }
            }
        }
    }
}

template <typename T>
void CopyArmComputeITensorData(const T* srcData, arm_compute::ITensor& dstTensor)
{
    // if MaxNumOfTensorDimensions is increased, this loop will need fixing
    static_assert(MaxNumOfTensorDimensions == 4, "Please update CopyArmComputeITensorData");
    {
        const arm_compute::ITensorInfo& info = *dstTensor.info();
        const arm_compute::TensorShape& shape = info.tensor_shape();
        uint8_t* const bufferPtr = dstTensor.buffer();
        uint32_t width = boost::numeric_cast<uint32_t>(shape[0]);
        uint32_t height = boost::numeric_cast<uint32_t>(shape[1]);
        uint32_t numChannels = boost::numeric_cast<uint32_t>(shape[2]);
        uint32_t numBatches = boost::numeric_cast<uint32_t>(shape[3]);

        for (unsigned int batchIndex = 0; batchIndex < numBatches; ++batchIndex)
        {
            for (unsigned int channelIndex = 0; channelIndex < numChannels; ++channelIndex)
            {
                for (unsigned int y = 0; y < height; ++y)
                {
                    // Copy one row from linear memory buffer to arm_compute tensor buffer
                    // A row is the largest contiguous region we can copy, as the tensor data may be using strides
                    memcpy(bufferPtr + GetTensorOffset(info, batchIndex, channelIndex, y, 0),
                           srcData + GetLinearBufferOffset(info, batchIndex, channelIndex, y, 0),
                           width * sizeof(T));
                }
            }
        }
    }
}

} // namespace armcomputetensorutils
} // namespace armnn
