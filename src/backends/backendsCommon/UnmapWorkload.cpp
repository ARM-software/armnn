//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/UnmapWorkload.hpp>

namespace armnn
{

UnmapWorkload::UnmapWorkload(const UnmapQueueDescriptor& descriptor,
                             const WorkloadInfo& info)
    : BaseWorkload<UnmapQueueDescriptor>(descriptor, info)
{
}

void UnmapWorkload::Execute() const
{
    m_Data.m_Inputs[0]->Unmap();
}

} //namespace armnn
