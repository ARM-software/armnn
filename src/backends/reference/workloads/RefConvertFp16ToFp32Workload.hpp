//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefConvertFp16ToFp32Workload : public Float16ToFloat32Workload<ConvertFp16ToFp32QueueDescriptor>
{
public:
    using Float16ToFloat32Workload<ConvertFp16ToFp32QueueDescriptor>::Float16ToFloat32Workload;
    virtual void Execute() const override;
};

} //namespace armnn
