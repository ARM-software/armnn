//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

#include "ScatterNd.hpp"

namespace armnn
{

    class RefScatterNdWorkload : public RefBaseWorkload<ScatterNdQueueDescriptor>
    {
    public:
        explicit RefScatterNdWorkload(const ScatterNdQueueDescriptor& descriptor,
                                      const WorkloadInfo& info);

        void Execute() const override;
        void ExecuteAsync(ExecutionData& executionData) override;

    private:
        void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;

    };

} // namespace armnn