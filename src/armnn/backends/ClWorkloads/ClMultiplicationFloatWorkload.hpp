//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

arm_compute::Status ClMultiplicationWorkloadValidate(const TensorInfo& input0,
                                                     const TensorInfo& input1,
                                                     const TensorInfo& output);

class ClMultiplicationFloatWorkload : public FloatWorkload<MultiplicationQueueDescriptor>
{
public:
    ClMultiplicationFloatWorkload(const MultiplicationQueueDescriptor& descriptor, const WorkloadInfo& info);

    using FloatWorkload<MultiplicationQueueDescriptor>::FloatWorkload;
    void Execute() const override;

private:
    mutable arm_compute::CLPixelWiseMultiplication   m_PixelWiseMultiplication;
};

} //namespace armnn



