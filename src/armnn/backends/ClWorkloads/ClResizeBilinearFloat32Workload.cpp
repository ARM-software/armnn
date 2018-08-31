//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClResizeBilinearFloat32Workload.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "backends/ClLayerSupport.hpp"
#include "backends/ArmComputeUtils.hpp"

namespace armnn
{

ClResizeBilinearFloat32Workload::ClResizeBilinearFloat32Workload(const ResizeBilinearQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info)
    : FloatWorkload<ResizeBilinearQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClResizeBilinearFloat32Workload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_ResizeBilinearLayer.configure(&input, &output, arm_compute::InterpolationPolicy::BILINEAR,
                                    arm_compute::BorderMode::REPLICATE, arm_compute::PixelValue(0.f),
                                    arm_compute::SamplingPolicy::TOP_LEFT);
};

void ClResizeBilinearFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClResizeBilinearFloat32Workload_Execute");
    m_ResizeBilinearLayer.run();
}


} //namespace armnn