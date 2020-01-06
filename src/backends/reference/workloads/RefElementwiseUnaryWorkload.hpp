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

class RefElementwiseUnaryWorkload : public BaseWorkload<ElementwiseUnaryQueueDescriptor>
{
public:
    using BaseWorkload<ElementwiseUnaryQueueDescriptor>::m_Data;

    RefElementwiseUnaryWorkload(const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info);
    void PostAllocationConfigure() override;
    void Execute() const override;

private:
    using InType  = float;
    using OutType = float;

    std::unique_ptr<Decoder<InType>>  m_Input;
    std::unique_ptr<Encoder<OutType>> m_Output;
};

} // namespace armnn
