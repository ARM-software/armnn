//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClStridedSliceWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <armnn/backends/TensorHandle.hpp>
#include <backendsCommon/WorkloadUtils.hpp>

#include <armnn/utility/NumericCast.hpp>

#include <cl/ClLayerSupport.hpp>
#include <cl/ClTensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status ClStridedSliceWorkloadValidate(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const StridedSliceDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);

    arm_compute::Coordinates starts;
    arm_compute::Coordinates ends;
    arm_compute::Coordinates strides;

    std::tie(starts, ends, strides) = SetClStridedSliceData(descriptor.m_Begin, descriptor.m_End, descriptor.m_Stride);

    auto numDimensions       = armnn::numeric_cast<int>(input.GetNumDimensions());
    int32_t begin_mask       = ConvertMaskToACLFormat(descriptor.m_BeginMask, numDimensions);
    int32_t end_mask         = ConvertMaskToACLFormat(descriptor.m_EndMask, numDimensions);
    int32_t shrink_axis_mask = ConvertMaskToACLFormat(descriptor.m_ShrinkAxisMask, numDimensions);

    return arm_compute::CLStridedSlice::validate(&aclInputInfo,
                                        &aclOutputInfo,
                                        starts,
                                        ends,
                                        strides,
                                        begin_mask,
                                        end_mask,
                                        shrink_axis_mask);
}

ClStridedSliceWorkload::ClStridedSliceWorkload(const StridedSliceQueueDescriptor& descriptor,
                                               const WorkloadInfo& info,
                                               const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<StridedSliceQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClStridedSliceWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClStridedSliceWorkload", 1, 1);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::Coordinates starts;
    arm_compute::Coordinates ends;
    arm_compute::Coordinates strides;

    std::tie(starts, ends, strides) = SetClStridedSliceData(m_Data.m_Parameters.m_Begin,
                                                            m_Data.m_Parameters.m_End,
                                                            m_Data.m_Parameters.m_Stride);

    auto numDimensions       = armnn::numeric_cast<int>(info.m_InputTensorInfos[0].GetNumDimensions());
    int32_t begin_mask       = ConvertMaskToACLFormat(m_Data.m_Parameters.m_BeginMask, numDimensions);
    int32_t end_mask         = ConvertMaskToACLFormat(m_Data.m_Parameters.m_EndMask, numDimensions);
    int32_t shrink_axis_mask = ConvertMaskToACLFormat(m_Data.m_Parameters.m_ShrinkAxisMask, numDimensions);

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClStridedSliceWorkload_configure");
        m_StridedSliceLayer.configure(clCompileContext,
                                      &input,
                                      &output,
                                      starts,
                                      ends,
                                      strides,
                                      begin_mask,
                                      end_mask,
                                      shrink_axis_mask);
    }
}

void ClStridedSliceWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClStridedSliceWorkload_Execute", this->GetGuid());
    RunClFunction(m_StridedSliceLayer, CHECK_LOCATION());
}

} //namespace armnn
