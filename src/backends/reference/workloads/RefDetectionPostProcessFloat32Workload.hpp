//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefDetectionPostProcessFloat32Workload : public Float32Workload<DetectionPostProcessQueueDescriptor>
{
public:
    explicit RefDetectionPostProcessFloat32Workload(const DetectionPostProcessQueueDescriptor& descriptor,
                                                    const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    std::unique_ptr<ScopedCpuTensorHandle> m_Anchors;
};

} //namespace armnn
