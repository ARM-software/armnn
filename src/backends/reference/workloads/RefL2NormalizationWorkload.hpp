//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

class RefL2NormalizationWorkload : public RefBaseWorkload<L2NormalizationQueueDescriptor>
{
public:
    explicit RefL2NormalizationWorkload(const L2NormalizationQueueDescriptor& descriptor,
                                        const WorkloadInfo& info);

    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)  override;
private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
};

} //namespace armnn
