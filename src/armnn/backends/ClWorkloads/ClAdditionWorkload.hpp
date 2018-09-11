//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

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

bool ClAdditionValidate(const TensorInfo& input0,
                        const TensorInfo& input1,
                        const TensorInfo& output,
                        std::string* reasonIfUnsupported);
} //namespace armnn
