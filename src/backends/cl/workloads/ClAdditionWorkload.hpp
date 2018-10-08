//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/Workload.hpp>

#include <arm_compute/runtime/CL/CLFunctions.h>

namespace armnn
{

template <armnn::DataType... dataTypes>
class ClAdditionWorkload : public TypedWorkload<AdditionQueueDescriptor, dataTypes...>
{
public:
    ClAdditionWorkload(const AdditionQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::CLArithmeticAddition m_Layer;
};

arm_compute::Status ClAdditionValidate(const TensorInfo& input0,
                                       const TensorInfo& input1,
                                       const TensorInfo& output);
} //namespace armnn
