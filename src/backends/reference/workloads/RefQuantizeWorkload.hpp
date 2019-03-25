//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn {

class RefQuantizeWorkload : public BaseWorkload<QuantizeQueueDescriptor>
{
public:
    RefQuantizeWorkload(const QuantizeQueueDescriptor& descriptor, const WorkloadInfo &info);
    void Execute() const override;

private:
    size_t m_NumElements;
    armnn::DataType m_TargetType;
    float m_Scale;
    int m_Offset;
};

} //namespace armnn