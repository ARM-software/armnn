//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefFullyConnectedFloat32Workload : public Float32Workload<FullyConnectedQueueDescriptor>
{
public:
    explicit RefFullyConnectedFloat32Workload(const FullyConnectedQueueDescriptor& descriptor,
                                                  const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    std::unique_ptr<ScopedCpuTensorHandle> m_Weight;
    std::unique_ptr<ScopedCpuTensorHandle> m_Bias;
};

} //namespace armnn
