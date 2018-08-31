//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonPooling2dUint8Workload.hpp"



namespace armnn
{

NeonPooling2dUint8Workload::NeonPooling2dUint8Workload(const Pooling2dQueueDescriptor& descriptor,
                                                       const WorkloadInfo& info)
    : NeonPooling2dBaseWorkload<armnn::DataType::QuantisedAsymm8>(descriptor, info, "NeonPooling2dUint8Workload")
{
}

void NeonPooling2dUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonPooling2dUint8Workload_Execute");
    m_PoolingLayer.run();
}

} //namespace armnn

