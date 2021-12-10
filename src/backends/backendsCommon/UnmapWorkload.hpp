//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/Workload.hpp>

namespace armnn
{

class UnmapWorkload : public BaseWorkload<UnmapQueueDescriptor>
{
public:
    UnmapWorkload(const UnmapQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;
};

} //namespace armnn
