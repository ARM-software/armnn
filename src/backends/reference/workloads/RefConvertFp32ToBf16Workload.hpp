//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefConvertFp32ToBf16Workload : public Float32ToBFloat16Workload<ConvertFp32ToBf16QueueDescriptor>
{
public:
    using Float32ToBFloat16Workload<ConvertFp32ToBf16QueueDescriptor>::Float32ToBFloat16Workload;
    virtual void Execute() const override;
};

} //namespace armnn
