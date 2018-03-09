//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "ClPooling2dUint8Workload_Execute");
    m_PoolingLayer.run();
}

} //namespace armnn


