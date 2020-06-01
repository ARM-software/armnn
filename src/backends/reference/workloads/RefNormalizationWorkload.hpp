//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefNormalizationWorkload : public BaseWorkload<NormalizationQueueDescriptor>
{
public:
    explicit RefNormalizationWorkload(const NormalizationQueueDescriptor& descriptor,
                                      const WorkloadInfo& info);

    virtual void Execute() const override;
};

} // namespace armnn
