//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/Workload.hpp>

#include "ClPooling2dBaseWorkload.hpp"

namespace armnn
{

class ClPooling2dUint8Workload : public ClPooling2dBaseWorkload<DataType::QuantisedAsymm8>
{
public:
    ClPooling2dUint8Workload(const Pooling2dQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

};

} //namespace armnn


