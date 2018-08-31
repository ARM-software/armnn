//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClPooling2dFloat32Workload.hpp"

namespace armnn
{

ClPooling2dFloat32Workload::ClPooling2dFloat32Workload(const Pooling2dQueueDescriptor& descriptor,
                                                       const WorkloadInfo& info)
    : ClPooling2dBaseWorkload<DataType::Float16, DataType::Float32>(descriptor, info, "ClPooling2dFloat32Workload")
{
}

void ClPooling2dFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClPooling2dFloat32Workload_Execute");
    m_PoolingLayer.run();
}

} //namespace armnn

