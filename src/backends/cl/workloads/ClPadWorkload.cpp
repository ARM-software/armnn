//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClPadWorkload.hpp"

#include <cl/ClTensorHandle.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <arm_compute/core/Types.h>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

ClPadWorkload::ClPadWorkload(const PadQueueDescriptor& descriptor,
                             const WorkloadInfo& info,
                             const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<PadQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClPadWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    this->m_Data.ValidateInputsOutputs("ClPadWorkload", 1, 1);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(this->m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(this->m_Data.m_Outputs[0])->GetTensor();

    std::vector<std::pair<unsigned int, unsigned int>> reversed_PadList(descriptor.m_Parameters.m_PadList.size());

    std::reverse_copy(std::begin(descriptor.m_Parameters.m_PadList),
            std::end(descriptor.m_Parameters.m_PadList),
            std::begin(reversed_PadList));

    arm_compute::PaddingList padList = static_cast<arm_compute::PaddingList>(reversed_PadList);

    arm_compute::PixelValue pixelValue = GetPixelValue(input.info(), descriptor.m_Parameters.m_PadValue);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClPadWorkload_configure");
        m_Layer.configure(clCompileContext,
                          &input,
                          &output,
                          padList,
                          pixelValue,
                          ConvertPaddingModeToAcl(descriptor.m_Parameters.m_PaddingMode));
    }
}

void ClPadWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClPadWorkload_Execute", this->GetGuid());
    RunClFunction(m_Layer, CHECK_LOCATION());
}

arm_compute::Status ClPadValidate(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const PadDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    std::vector<std::pair<unsigned int, unsigned int>> reversed_PadList(descriptor.m_PadList.size());

    std::reverse_copy(std::begin(descriptor.m_PadList),
                      std::end(descriptor.m_PadList),
                      std::begin(reversed_PadList));

    arm_compute::PaddingList padList = static_cast<arm_compute::PaddingList>(reversed_PadList);

    // PixelValue is currently unused when validating, but it's required to pass in PaddingMode.
    arm_compute::PixelValue pixelValue = GetPixelValue(&aclInputInfo, descriptor.m_PadValue);
    const arm_compute::Status aclStatus =
            arm_compute::CLPadLayer::validate(&aclInputInfo,
                                              &aclOutputInfo,
                                              padList,
                                              pixelValue,
                                              ConvertPaddingModeToAcl(descriptor.m_PaddingMode));

    return aclStatus;
}

} // namespace armnn
