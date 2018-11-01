//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClResizeBilinearFloatWorkload.hpp"
#include <cl/ClTensorHandle.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include "ClWorkloadUtils.hpp"

using namespace armnn::armcomputetensorutils;

namespace armnn
{

ClResizeBilinearFloatWorkload::ClResizeBilinearFloatWorkload(const ResizeBilinearQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info)
    : FloatWorkload<ResizeBilinearQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClResizeBilinearFloatWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout.GetDataLayout());
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    m_ResizeBilinearLayer.configure(&input, &output, arm_compute::InterpolationPolicy::BILINEAR,
                                    arm_compute::BorderMode::REPLICATE, arm_compute::PixelValue(0.f),
                                    arm_compute::SamplingPolicy::TOP_LEFT);
};

void ClResizeBilinearFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClResizeBilinearFloatWorkload_Execute");
    RunClFunction(m_ResizeBilinearLayer, CHECK_LOCATION());
}

} //namespace armnn
