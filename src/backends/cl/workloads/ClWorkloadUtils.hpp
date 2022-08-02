//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <BFloat16.hpp>
#include <Half.hpp>

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <cl/OpenClTimer.hpp>
#include <armnn/backends/TensorHandle.hpp>

#include <armnn/Utils.hpp>

#include <arm_compute/runtime/CL/CLTensor.h>
#include <arm_compute/runtime/IFunction.h>

#include <sstream>

#define ARMNN_SCOPED_PROFILING_EVENT_CL(name) \
    ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(armnn::Compute::GpuAcc, \
                                                  armnn::EmptyOptional(), \
                                                  name, \
                                                  armnn::OpenClTimer(), \
                                                  armnn::WallClockTimer())

#define ARMNN_SCOPED_PROFILING_EVENT_CL_GUID(name, guid) \
    ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(armnn::Compute::GpuAcc, \
                                                  guid, \
                                                  name, \
                                                  armnn::OpenClTimer(), \
                                                  armnn::WallClockTimer())

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
void CopyArmComputeClTensorData(arm_compute::CLTensor& dstTensor, const T* srcData)
{
    {
        ARMNN_SCOPED_PROFILING_EVENT_CL("MapClTensorForWriting");
        dstTensor.map(true);
    }

    {
        ARMNN_SCOPED_PROFILING_EVENT_CL("CopyToClTensor");
        armcomputetensorutils::CopyArmComputeITensorData<T>(srcData, dstTensor);
    }

    dstTensor.unmap();
}

inline auto SetClStridedSliceData(const std::vector<int>& m_begin,
                                  const std::vector<int>& m_end,
                                  const std::vector<int>& m_stride)
{
    arm_compute::Coordinates starts;
    arm_compute::Coordinates ends;
    arm_compute::Coordinates strides;

    unsigned int num_dims = static_cast<unsigned int>(m_begin.size());

    for (unsigned int i = 0; i < num_dims; i++) {
        unsigned int revertedIndex = num_dims - i - 1;

        starts.set(i, static_cast<int>(m_begin[revertedIndex]));
        ends.set(i, static_cast<int>(m_end[revertedIndex]));
        strides.set(i, static_cast<int>(m_stride[revertedIndex]));
    }

    return std::make_tuple(starts, ends, strides);
}

inline auto SetClSliceData(const std::vector<unsigned int>& m_begin,
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

inline void InitializeArmComputeClTensorData(arm_compute::CLTensor& clTensor,
                                             const ConstTensorHandle* handle)
{
    ARMNN_ASSERT(handle);

    armcomputetensorutils::InitialiseArmComputeTensorEmpty(clTensor);
    switch(handle->GetTensorInfo().GetDataType())
    {
        case DataType::Float16:
            CopyArmComputeClTensorData(clTensor, handle->GetConstTensor<armnn::Half>());
            break;
        case DataType::Float32:
            CopyArmComputeClTensorData(clTensor, handle->GetConstTensor<float>());
            break;
        case DataType::QAsymmU8:
            CopyArmComputeClTensorData(clTensor, handle->GetConstTensor<uint8_t>());
            break;
        case DataType::QAsymmS8:
        case DataType::QSymmS8:
            CopyArmComputeClTensorData(clTensor, handle->GetConstTensor<int8_t>());
            break;
        case DataType::QSymmS16:
            CopyArmComputeClTensorData(clTensor, handle->GetConstTensor<int16_t>());
            break;
        case DataType::Signed32:
            CopyArmComputeClTensorData(clTensor, handle->GetConstTensor<int32_t>());
            break;
        case DataType::BFloat16:
            CopyArmComputeClTensorData(clTensor, handle->GetConstTensor<armnn::BFloat16>());
            break;
        default:
            // Throw exception; assertion not called in release build.
            throw Exception("Unexpected tensor type during InitializeArmComputeClTensorData().");
    }
};

inline RuntimeException WrapClError(const cl::Error& clError, const CheckLocation& location)
{
    std::stringstream message;
    message << "CL error: " << clError.what() << ". Error code: " << clError.err();

    return RuntimeException(message.str(), location);
}

inline void RunClFunction(arm_compute::IFunction& function, const CheckLocation& location)
{
    try
    {
        function.run();
    }
    catch (cl::Error& error)
    {
        throw WrapClError(error, location);
    }
}

template <typename DataType, typename PayloadType>
DataType* GetOutputTensorData(unsigned int idx, const PayloadType& data)
{
    ITensorHandle* tensorHandle = data.m_Outputs[idx];
    return reinterpret_cast<DataType*>(tensorHandle->Map());
}

} //namespace armnn
