//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonPooling2dFloatWorkload.hpp"



namespace armnn
{

NeonPooling2dFloatWorkload::NeonPooling2dFloatWorkload(const Pooling2dQueueDescriptor& descriptor,
                                                       const WorkloadInfo& info)
    : NeonPooling2dBaseWorkload<armnn::DataType::Float16, armnn::DataType::Float32>(descriptor, info,
                                                                                    "NeonPooling2dFloatWorkload")
{
}

void NeonPooling2dFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonPooling2dFloatWorkload_Execute");
    m_PoolingLayer.run();
}

} //namespace armnn

