//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

arm_compute::Status ClMultiplicationWorkloadValidate(const TensorInfo& input0,
                                                     const TensorInfo& input1,
                                                     const TensorInfo& output);

class ClMultiplicationFloat32Workload : public FloatWorkload<MultiplicationQueueDescriptor>
{
public:
    ClMultiplicationFloat32Workload(const MultiplicationQueueDescriptor& descriptor, const WorkloadInfo& info);

    using FloatWorkload<MultiplicationQueueDescriptor>::FloatWorkload;
    void Execute() const override;

private:
    mutable arm_compute::CLPixelWiseMultiplication   m_PixelWiseMultiplication;
};

} //namespace armnn



