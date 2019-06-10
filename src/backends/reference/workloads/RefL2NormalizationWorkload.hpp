//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefL2NormalizationWorkload : public BaseWorkload<L2NormalizationQueueDescriptor>
{
public:
    explicit RefL2NormalizationWorkload(const L2NormalizationQueueDescriptor& descriptor,
                                        const WorkloadInfo& info);

    void Execute() const override;
};

} //namespace armnn
