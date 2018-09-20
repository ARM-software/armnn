//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClPooling2dFloatWorkload.hpp"

#include "ClWorkloadUtils.hpp"

namespace armnn
{

ClPooling2dFloatWorkload::ClPooling2dFloatWorkload(const Pooling2dQueueDescriptor& descriptor,
                                                       const WorkloadInfo& info)
    : ClPooling2dBaseWorkload<DataType::Float16, DataType::Float32>(descriptor, info, "ClPooling2dFloatWorkload")
{
}

void ClPooling2dFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClPooling2dFloatWorkload_Execute");
    m_PoolingLayer.run();
}

} //namespace armnn

