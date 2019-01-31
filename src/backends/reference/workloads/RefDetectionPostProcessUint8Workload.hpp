//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefDetectionPostProcessUint8Workload : public Uint8ToFloat32Workload<DetectionPostProcessQueueDescriptor>
{
public:
    explicit RefDetectionPostProcessUint8Workload(const DetectionPostProcessQueueDescriptor& descriptor,
                                                  const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    std::unique_ptr<ScopedCpuTensorHandle> m_Anchors;
};

} //namespace armnn
