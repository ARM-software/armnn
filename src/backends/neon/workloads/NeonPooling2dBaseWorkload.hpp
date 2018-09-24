//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/neon/workloads/NeonWorkloadUtils.hpp>

namespace armnn
{

arm_compute::Status NeonPooling2dWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const Pooling2dDescriptor& descriptor);

// Base class template providing an implementation of the Pooling2d layer common to all data types.
template <armnn::DataType... dataTypes>
class NeonPooling2dBaseWorkload : public TypedWorkload<Pooling2dQueueDescriptor, dataTypes...>
{
public:
    using TypedWorkload<Pooling2dQueueDescriptor, dataTypes...>::m_Data;

    NeonPooling2dBaseWorkload(const Pooling2dQueueDescriptor& descriptor, const WorkloadInfo& info,
                              const std::string& name);

protected:
    mutable arm_compute::NEPoolingLayer m_PoolingLayer;
};


} //namespace armnn





