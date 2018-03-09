//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

namespace armnn
{
class NeonAdditionFloat32Workload : public Float32Workload<AdditionQueueDescriptor>
{
public:
    NeonAdditionFloat32Workload(const AdditionQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEArithmeticAddition m_AddLayer;
};

} //namespace armnn



