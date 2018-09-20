//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "backends/Workload.hpp"

#include <arm_compute/runtime/CL/CLFunctions.h>

namespace armnn
{

template <armnn::DataType... dataTypes>
class ClSubtractionWorkload : public TypedWorkload<SubtractionQueueDescriptor, dataTypes...>
{
public:
    ClSubtractionWorkload(const SubtractionQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::CLArithmeticSubtraction m_Layer;
};

bool ClSubtractionValidate(const TensorInfo& input0,
                           const TensorInfo& input1,
                           const TensorInfo& output,
                           std::string* reasonIfUnsupported);
} //namespace armnn
