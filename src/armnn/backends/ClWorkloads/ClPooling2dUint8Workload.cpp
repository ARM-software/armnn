//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClPooling2dUint8Workload.hpp"

namespace armnn
{

ClPooling2dUint8Workload::ClPooling2dUint8Workload(const Pooling2dQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info)
    : ClPooling2dBaseWorkload<DataType::QuantisedAsymm8>(descriptor, info, "ClPooling2dUint8Workload")
{
}

void ClPooling2dUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClPooling2dUint8Workload_Execute");
    m_PoolingLayer.run();
}

} //namespace armnn


