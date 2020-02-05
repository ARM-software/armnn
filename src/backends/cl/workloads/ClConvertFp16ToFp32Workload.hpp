//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLDepthConvertLayer.h>

namespace armnn
{

class ClConvertFp16ToFp32Workload : public Float16ToFloat32Workload<ConvertFp16ToFp32QueueDescriptor>
{
public:

    ClConvertFp16ToFp32Workload(const ConvertFp16ToFp32QueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLDepthConvertLayer m_Layer;
};

arm_compute::Status ClConvertFp16ToFp32WorkloadValidate(const TensorInfo& input, const TensorInfo& output);

} //namespace armnn
