//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{
class ClMultiplicationFloat32Workload : public Float32Workload<MultiplicationQueueDescriptor>
{
public:
    ClMultiplicationFloat32Workload(const MultiplicationQueueDescriptor& descriptor, const WorkloadInfo& info);

    using Float32Workload<MultiplicationQueueDescriptor>::Float32Workload;
    void Execute() const override;

private:
    mutable arm_compute::CLPixelWiseMultiplication   m_PixelWiseMultiplication;
};

} //namespace armnn



