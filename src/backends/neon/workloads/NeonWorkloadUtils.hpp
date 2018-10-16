//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <backends/Workload.hpp>
#include <backends/aclCommon/ArmComputeTensorUtils.hpp>
#include <backends/neon/NeonTensorHandle.hpp>
#include <backends/neon/NeonTimer.hpp>
#include <backends/CpuTensorHandle.hpp>
#include <arm_compute/runtime/NEON/NEFunctions.h>

#include <Half.hpp>

#define ARMNN_SCOPED_PROFILING_EVENT_NEON(name) \
    ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(armnn::Compute::CpuAcc, \
                                                  name, \
                                                  armnn::NeonTimer(), \
                                                  armnn::WallClockTimer())

using namespace armnn::armcomputetensorutils;

namespace armnn
{

template <typename T>
void CopyArmComputeTensorData(arm_compute::Tensor& dstTensor, const T* srcData)
{
    InitialiseArmComputeTensorEmpty(dstTensor);
    CopyArmComputeITensorData(srcData, dstTensor);
}

inline void InitializeArmComputeTensorData(arm_compute::Tensor& tensor,
                                           const ConstCpuTensorHandle* handle)
{
    BOOST_ASSERT(handle);

    switch(handle->GetTensorInfo().GetDataType())
    {
        case DataType::Float16:
            CopyArmComputeTensorData(tensor, handle->GetConstTensor<armnn::Half>());
            break;
        case DataType::Float32:
            CopyArmComputeTensorData(tensor, handle->GetConstTensor<float>());
            break;
        case DataType::QuantisedAsymm8:
            CopyArmComputeTensorData(tensor, handle->GetConstTensor<uint8_t>());
            break;
        case DataType::Signed32:
            CopyArmComputeTensorData(tensor, handle->GetConstTensor<int32_t>());
            break;
        default:
            BOOST_ASSERT_MSG(false, "Unexpected tensor type.");
    }
};

} //namespace armnn
