//
// Copyright © 2020 Samsung Electronics Co Ltd and Contributors. All rights reserved.
// Copyright © 2021-2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

class RefReduceWorkload : public RefBaseWorkload<ReduceQueueDescriptor>
{
public:
    explicit RefReduceWorkload(const ReduceQueueDescriptor& descriptor,
                               const WorkloadInfo& info);

    void Execute() const override;
    void ExecuteAsync(ExecutionData& executionData)  override;
private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
};

} //namespace armnn
