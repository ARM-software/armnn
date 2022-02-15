//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

class RefComparisonWorkload : public RefBaseWorkload<ComparisonQueueDescriptor>
{
public:
    using RefBaseWorkload<ComparisonQueueDescriptor>::m_Data;

    RefComparisonWorkload(const ComparisonQueueDescriptor& descriptor, const WorkloadInfo& info);
    void PostAllocationConfigure() override;
    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)  override;

private:
    void PostAllocationConfigure(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs);
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
    using InType  = float;
    using OutType = bool;

    std::unique_ptr<Decoder<InType>>  m_Input0;
    std::unique_ptr<Decoder<InType>>  m_Input1;
    std::unique_ptr<Encoder<OutType>> m_Output;
};

} // namespace armnn
