//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefComparisonWorkload : public BaseWorkload<ComparisonQueueDescriptor>
{
public:
    using BaseWorkload<ComparisonQueueDescriptor>::m_Data;

    RefComparisonWorkload(const ComparisonQueueDescriptor& descriptor, const WorkloadInfo& info);
    void PostAllocationConfigure() override;
    void Execute() const override;

private:
    using InType  = float;
    using OutType = bool;

    std::unique_ptr<Decoder<InType>>  m_Input0;
    std::unique_ptr<Decoder<InType>>  m_Input1;
    std::unique_ptr<Encoder<OutType>> m_Output;
};

} // namespace armnn
