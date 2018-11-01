//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefConvolution2dUint8Workload : public Uint8Workload<Convolution2dQueueDescriptor>
{
public:
    explicit RefConvolution2dUint8Workload(const Convolution2dQueueDescriptor& descriptor,
                                             const WorkloadInfo& info);

    virtual void Execute() const override;

private:
    std::unique_ptr<ScopedCpuTensorHandle> m_Weight;
    std::unique_ptr<ScopedCpuTensorHandle> m_Bias;

};

} //namespace armnn
