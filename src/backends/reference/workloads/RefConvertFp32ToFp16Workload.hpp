//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/Workload.hpp>
#include <backends/WorkloadData.hpp>

namespace armnn
{

class RefConvertFp32ToFp16Workload : public Float32ToFloat16Workload<ConvertFp32ToFp16QueueDescriptor>
{
public:
    using Float32ToFloat16Workload<ConvertFp32ToFp16QueueDescriptor>::Float32ToFloat16Workload;
    virtual void Execute() const override;
};

} //namespace armnn
