//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefConvertBf16ToFp32Workload : public BFloat16ToFloat32Workload<ConvertBf16ToFp32QueueDescriptor>
{
public:
    using BFloat16ToFloat32Workload<ConvertBf16ToFp32QueueDescriptor>::BFloat16ToFloat32Workload;
    virtual void Execute() const override;
};

} //namespace armnn
