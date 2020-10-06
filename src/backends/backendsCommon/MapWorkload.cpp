//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/MapWorkload.hpp>

namespace armnn
{

MapWorkload::MapWorkload(const MapQueueDescriptor& descriptor,
                         const WorkloadInfo& info)
    : BaseWorkload<MapQueueDescriptor>(descriptor, info)
{
}

void MapWorkload::Execute() const
{
    m_Data.m_Inputs[0]->Map();
}

} //namespace armnn
