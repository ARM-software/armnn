//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonPooling2dFloat32Workload.hpp"



namespace armnn
{

NeonPooling2dFloat32Workload::NeonPooling2dFloat32Workload(const Pooling2dQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info)
    : NeonPooling2dBaseWorkload<armnn::DataType::Float16, armnn::DataType::Float32>(descriptor, info,
                                                                                    "NeonPooling2dFloat32Workload")
{
}

void NeonPooling2dFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonPooling2dFloat32Workload_Execute");
    m_PoolingLayer.run();
}

} //namespace armnn

