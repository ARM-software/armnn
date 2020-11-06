//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefLogicalUnaryWorkload : public BaseWorkload<ElementwiseUnaryQueueDescriptor>
{
public:
    using BaseWorkload<ElementwiseUnaryQueueDescriptor>::m_Data;

    RefLogicalUnaryWorkload(const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info);
    void PostAllocationConfigure() override;
    virtual void Execute() const override;

private:
    using InType  = bool;
    using OutType = bool;

    std::unique_ptr<Decoder<InType>>  m_Input;
    std::unique_ptr<Encoder<OutType>> m_Output;
};

} // namespace armnn
