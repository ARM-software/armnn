//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/Workload.hpp>

namespace armnn
{

class MapWorkload : public BaseWorkload<MapQueueDescriptor>
{
public:
    MapWorkload(const MapQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;
};

} //namespace armnn
