//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"

namespace armnn
{

class RefDepthwiseConvolution2dUint8Workload : public Uint8Workload<DepthwiseConvolution2dQueueDescriptor>
{
public:
    explicit RefDepthwiseConvolution2dUint8Workload(const DepthwiseConvolution2dQueueDescriptor& descriptor,
                                           const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    std::unique_ptr<ScopedCpuTensorHandle> m_Weight;
    std::unique_ptr<ScopedCpuTensorHandle> m_Bias;
};

} //namespace armnn
