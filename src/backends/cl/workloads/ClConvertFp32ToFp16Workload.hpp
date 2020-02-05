//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLDepthConvertLayer.h>

namespace armnn
{

class ClConvertFp32ToFp16Workload : public Float32ToFloat16Workload<ConvertFp32ToFp16QueueDescriptor>
{
public:

    ClConvertFp32ToFp16Workload(const ConvertFp32ToFp16QueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLDepthConvertLayer m_Layer;
};

arm_compute::Status ClConvertFp32ToFp16WorkloadValidate(const TensorInfo& input, const TensorInfo& output);

} //namespace armnn
