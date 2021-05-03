//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClReduceWorkload.hpp"

#include <cl/ClTensorHandle.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClReduceWorkloadValidate(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const ReduceDescriptor& desc)
{
    const arm_compute::TensorInfo aclInputInfo  = armcomputetensorutils::BuildArmComputeTensorInfo(input);

    arm_compute::Coordinates coords = BuildArmComputeReductionCoordinates(aclInputInfo.num_dimensions(),
                                                                          input.GetNumDimensions(),
                                                                          desc.m_vAxis);

    // As ACL only support one axis, validate the layer for each axis if more than one is present.
    if (!desc.m_vAxis.empty() && desc.m_vAxis.size() > 1)
    {
        arm_compute::Status status;

        for (unsigned int i = 0; i != desc.m_vAxis.size(); ++i)
        {
            TensorInfo inputToModify = input;
            std::vector<uint32_t> singleAxis(1, desc.m_vAxis[i]);

            // Calculate the output shape using the input shape for a single axis.
            // Currently the output TensorInfo inferred will be reduced upon multiple axis
            // which will fail validation as only one axis is supported.
            const TensorShape& reducedShape = ComputeReductionTensorShape(inputToModify, singleAxis, desc.m_KeepDims);
            inputToModify.SetShape(reducedShape);

            const arm_compute::TensorInfo aclOutputInfoModified =
                    armcomputetensorutils::BuildArmComputeTensorInfo(inputToModify);

            status = arm_compute::CLReductionOperation::validate(&aclInputInfo,
                                                                 &aclOutputInfoModified,
                                                                 static_cast<unsigned int>(coords[i]),
                                                                 ConvertReductionOperationToAcl(desc),
                                                                 desc.m_KeepDims);
            if (!status)
            {
                break;
            }
        }
        return status;
    }
    else
    {
        const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

        return arm_compute::CLReductionOperation::validate(&aclInputInfo,
                                                           &aclOutputInfo,
                                                           static_cast<unsigned int>(coords[0]),
                                                           ConvertReductionOperationToAcl(desc),
                                                           desc.m_KeepDims);
    }
}

ClReduceWorkload::ClReduceWorkload(const ReduceQueueDescriptor& descriptor, const WorkloadInfo& info)
    : BaseWorkload<ReduceQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClReduceWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::Coordinates coords = BuildArmComputeReductionCoordinates(input.info()->num_dimensions(),
                                                                          info.m_InputTensorInfos[0].GetNumDimensions(),
                                                                          m_Data.m_Parameters.m_vAxis);
    m_Layer.configure(&input,
                      &output,
                      static_cast<unsigned int>(coords[0]),
                      ConvertReductionOperationToAcl(m_Data.m_Parameters),
                      m_Data.m_Parameters.m_KeepDims);
}

void ClReduceWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClReduceWorkload_Execute");
    m_Layer.run();
}

} //namespace armnn
