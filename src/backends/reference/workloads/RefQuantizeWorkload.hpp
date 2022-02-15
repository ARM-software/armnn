//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>
#include "Decoders.hpp"
#include "Encoders.hpp"

namespace armnn {

class RefQuantizeWorkload : public RefBaseWorkload<QuantizeQueueDescriptor>
{
public:
    RefQuantizeWorkload(const QuantizeQueueDescriptor& descriptor, const WorkloadInfo &info);
    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)  override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;

    size_t m_NumElements;
};

} //namespace armnn