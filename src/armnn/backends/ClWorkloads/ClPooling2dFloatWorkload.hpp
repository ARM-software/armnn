//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"
#include "backends/ClWorkloads//ClPooling2dBaseWorkload.hpp"

namespace armnn
{
class ClPooling2dFloatWorkload : public ClPooling2dBaseWorkload<DataType::Float16, DataType::Float32>
{
public:
    ClPooling2dFloatWorkload(const Pooling2dQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

};

} //namespace armnn
