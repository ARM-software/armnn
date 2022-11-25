//
// Copyright © 2017,2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/Workload.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <neon/NeonTensorHandle.hpp>
#include <neon/NeonTimer.hpp>
#include <armnn/backends/TensorHandle.hpp>

#include <armnn/Utils.hpp>

#include <Half.hpp>

#define ARMNN_SCOPED_PROFILING_EVENT_NEON(name) \
    ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(armnn::Compute::CpuAcc, \
                                                  armnn::EmptyOptional(), \
                                                  name, \
                                                  armnn::NeonTimer(), \
                                                  armnn::WallClockTimer())

#define ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID(name, guid) \
    ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(armnn::Compute::CpuAcc, \
                                                  guid, \
                                                  name, \
                                                  armnn::NeonTimer(), \
                                                  armnn::WallClockTimer())

using namespace armnn::armcomputetensorutils;

namespace armnn
{

inline std::string GetConvolutionMethodString(arm_compute::ConvolutionMethod& convolutionMethod)
{
    switch (convolutionMethod)
    {
        case arm_compute::ConvolutionMethod::FFT:
            return "FFT";
        case arm_compute::ConvolutionMethod::DIRECT:
            return "Direct";
        case arm_compute::ConvolutionMethod::GEMM:
            return "GEMM";
        case arm_compute::ConvolutionMethod::WINOGRAD:
            return "Winograd";
        default:
            return "Unknown";
    }
}

template <typename T>
void CopyArmComputeTensorData(arm_compute::Tensor& dstTensor, const T* srcData)
{
    InitialiseArmComputeTensorEmpty(dstTensor);
    CopyArmComputeITensorData(srcData, dstTensor);
}

inline void InitializeArmComputeTensorData(arm_compute::Tensor& tensor,
                                           TensorInfo tensorInfo,
                                           const ITensorHandle* handle)
{
    ARMNN_ASSERT(handle);

    switch(tensorInfo.GetDataType())
    {
        case DataType::Float16:
            CopyArmComputeTensorData(tensor, reinterpret_cast<const armnn::Half*>(handle->Map()));
            break;
        case DataType::Float32:
            CopyArmComputeTensorData(tensor, reinterpret_cast<const float*>(handle->Map()));
            break;
        case DataType::QAsymmU8:
            CopyArmComputeTensorData(tensor, reinterpret_cast<const uint8_t*>(handle->Map()));
            break;
        case DataType::QSymmS8:
        case DataType::QAsymmS8:
            CopyArmComputeTensorData(tensor, reinterpret_cast<const int8_t*>(handle->Map()));
            break;
        case DataType::Signed32:
            CopyArmComputeTensorData(tensor, reinterpret_cast<const int32_t*>(handle->Map()));
            break;
        case DataType::QSymmS16:
            CopyArmComputeTensorData(tensor, reinterpret_cast<const int16_t*>(handle->Map()));
            break;
        case DataType::BFloat16:
            CopyArmComputeTensorData(tensor, reinterpret_cast<const armnn::BFloat16*>(handle->Map()));
            break;
        default:
            // Throw exception; assertion not called in release build.
            throw Exception("Unexpected tensor type during InitializeArmComputeTensorData().");
    }
};

inline void InitializeArmComputeTensorData(arm_compute::Tensor& tensor,
                                           const ConstTensorHandle* handle)
{
    ARMNN_ASSERT(handle);

    switch(handle->GetTensorInfo().GetDataType())
    {
        case DataType::Float16:
            CopyArmComputeTensorData(tensor, handle->GetConstTensor<armnn::Half>());
            break;
        case DataType::Float32:
            CopyArmComputeTensorData(tensor, handle->GetConstTensor<float>());
            break;
        case DataType::QAsymmU8:
            CopyArmComputeTensorData(tensor, handle->GetConstTensor<uint8_t>());
            break;
        case DataType::QSymmS8:
        case DataType::QAsymmS8:
            CopyArmComputeTensorData(tensor, handle->GetConstTensor<int8_t>());
            break;
        case DataType::Signed32:
            CopyArmComputeTensorData(tensor, handle->GetConstTensor<int32_t>());
            break;
        case DataType::QSymmS16:
            CopyArmComputeTensorData(tensor, handle->GetConstTensor<int16_t>());
            break;
        case DataType::BFloat16:
            CopyArmComputeTensorData(tensor, handle->GetConstTensor<armnn::BFloat16>());
            break;
        default:
            // Throw exception; assertion not called in release build.
            throw Exception("Unexpected tensor type during InitializeArmComputeTensorData().");
    }
};

inline auto SetNeonStridedSliceData(const std::vector<int>& m_begin,
                                    const std::vector<int>& m_end,
                                    const std::vector<int>& m_stride)
{
    arm_compute::Coordinates starts;
    arm_compute::Coordinates ends;
    arm_compute::Coordinates strides;

    unsigned int num_dims = static_cast<unsigned int>(m_begin.size());

    for (unsigned int i = 0; i < num_dims; i++)
    {
        unsigned int revertedIndex = num_dims - i - 1;

        starts.set(i, static_cast<int>(m_begin[revertedIndex]));
        ends.set(i, static_cast<int>(m_end[revertedIndex]));
        strides.set(i, static_cast<int>(m_stride[revertedIndex]));
    }

    return std::make_tuple(starts, ends, strides);
}

inline auto SetNeonSliceData(const std::vector<unsigned int>& m_begin,
                             const std::vector<unsigned int>& m_size)
{
    // This function must translate the size vector given to an end vector
    // expected by the ACL NESlice workload
    arm_compute::Coordinates starts;
    arm_compute::Coordinates ends;

    unsigned int num_dims = static_cast<unsigned int>(m_begin.size());

    // For strided slices, we have the relationship size = (end - begin) / stride
    // For slice, we assume stride to be a vector of all ones, yielding the formula
    // size = (end - begin) therefore we know end = size + begin
    for (unsigned int i = 0; i < num_dims; i++)
    {
        unsigned int revertedIndex = num_dims - i - 1;

        starts.set(i, static_cast<int>(m_begin[revertedIndex]));
        ends.set(i, static_cast<int>(m_begin[revertedIndex] + m_size[revertedIndex]));
    }

    return std::make_tuple(starts, ends);
}

template <typename DataType, typename PayloadType>
DataType* GetOutputTensorData(unsigned int idx, const PayloadType& data)
{
    ITensorHandle* tensorHandle = data.m_Outputs[idx];
    return reinterpret_cast<DataType*>(tensorHandle->Map());
}

} //namespace armnn
