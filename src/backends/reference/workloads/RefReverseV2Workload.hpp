//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

#include "ReverseV2Impl.hpp"

namespace armnn
{

    class RefReverseV2Workload : public RefBaseWorkload<ReverseV2QueueDescriptor>
    {
    public:
        explicit RefReverseV2Workload(const ReverseV2QueueDescriptor& descriptor,
                                      const WorkloadInfo& info);

        void Execute() const override;
        void ExecuteAsync(ExecutionData& executionData) override;

    private:
        void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;

    };

} // namespace armnn