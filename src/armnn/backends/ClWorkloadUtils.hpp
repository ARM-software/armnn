//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "Workload.hpp"
#include <arm_compute/core/CL/OpenCL.h>
#include <arm_compute/runtime/CL/CLFunctions.h>
#include <arm_compute/runtime/SubTensor.h>
#include "ArmComputeTensorUtils.hpp"

namespace armnn
{

template <typename T>
void CopyArmComputeClTensorData(const T* srcData, arm_compute::CLTensor& dstTensor)
{
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "MapClTensorForWriting");
        dstTensor.map(true);
    }

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "CopyToClTensor");
        armcomputetensorutils::CopyArmComputeITensorData<T>(srcData, dstTensor);
    }

    dstTensor.unmap();
}

template <typename T>
void InitialiseArmComputeClTensorData(arm_compute::CLTensor& clTensor, const T* data)
{
    armcomputetensorutils::InitialiseArmComputeTensorEmpty(clTensor);
    CopyArmComputeClTensorData<T>(data, clTensor);
}

} //namespace armnn
