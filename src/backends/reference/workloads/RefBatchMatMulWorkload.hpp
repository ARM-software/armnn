//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

#include "BatchMatMulImpl.hpp"

namespace armnn
{

class RefBatchMatMulWorkload : public RefBaseWorkload<BatchMatMulQueueDescriptor>
{
public:
    explicit RefBatchMatMulWorkload(const BatchMatMulQueueDescriptor& descriptor,
                                    const WorkloadInfo& info);

    void Execute() const override;
    void ExecuteAsync(ExecutionData& executionData) override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;

};

} // namespace armnn