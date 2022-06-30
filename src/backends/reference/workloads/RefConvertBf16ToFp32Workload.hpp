//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

class RefConvertBf16ToFp32Workload : public BFloat16ToFloat32Workload<ConvertBf16ToFp32QueueDescriptor>
{
public:
    using BFloat16ToFloat32Workload<ConvertBf16ToFp32QueueDescriptor>::BFloat16ToFloat32Workload;
    void Execute() const override;
    void ExecuteAsync(ExecutionData& executionData)  override;
private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
};

} //namespace armnn
