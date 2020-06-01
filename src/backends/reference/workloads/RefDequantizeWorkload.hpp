//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

namespace armnn
{

class RefDequantizeWorkload : public BaseWorkload<DequantizeQueueDescriptor>
{
public:
    using BaseWorkload<DequantizeQueueDescriptor>::m_Data;
    using BaseWorkload<DequantizeQueueDescriptor>::BaseWorkload;

    void Execute() const override;
};

} // namespace armnn
