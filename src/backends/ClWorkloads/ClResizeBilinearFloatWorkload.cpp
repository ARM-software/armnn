//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClResizeBilinearFloatWorkload.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "backends/ClLayerSupport.hpp"
#include "backends/ArmComputeUtils.hpp"

#include "ClWorkloadUtils.hpp"

namespace armnn
{

ClResizeBilinearFloatWorkload::ClResizeBilinearFloatWorkload(const ResizeBilinearQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info)
    : FloatWorkload<ResizeBilinearQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClResizeBilinearFloatWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_ResizeBilinearLayer.configure(&input, &output, arm_compute::InterpolationPolicy::BILINEAR,
                                    arm_compute::BorderMode::REPLICATE, arm_compute::PixelValue(0.f),
                                    arm_compute::SamplingPolicy::TOP_LEFT);
};

void ClResizeBilinearFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClResizeBilinearFloatWorkload_Execute");
    m_ResizeBilinearLayer.run();
}


} //namespace armnn
