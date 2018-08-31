//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

class ClResizeBilinearFloat32Workload : public FloatWorkload<ResizeBilinearQueueDescriptor>
{
public:
    ClResizeBilinearFloat32Workload(const ResizeBilinearQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::CLScale m_ResizeBilinearLayer;
};

} //namespace armnn