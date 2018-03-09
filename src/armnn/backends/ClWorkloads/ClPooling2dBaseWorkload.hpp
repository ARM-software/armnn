//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

arm_compute::Status ClPooling2dWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const Pooling2dDescriptor& descriptor);

// Base class template providing an implementation of the Pooling2d layer common to all data types
template <armnn::DataType dataType>
class ClPooling2dBaseWorkload : public TypedWorkload<Pooling2dQueueDescriptor, dataType>
{
public:
    using TypedWorkload<Pooling2dQueueDescriptor, dataType>::m_Data;

    ClPooling2dBaseWorkload(const Pooling2dQueueDescriptor& descriptor, const WorkloadInfo& info, 
                            const std::string& name);

protected:
    mutable arm_compute::CLPoolingLayer m_PoolingLayer;
};

} //namespace armnn
