//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClMeanWorkload.hpp"

#include <backends/cl/ClTensorHandle.hpp>
#include <backends/aclCommon/ArmComputeTensorUtils.hpp>

#include "ClWorkloadUtils.hpp"

namespace
{

void ConvertArmnnAxesToAclCoordinates(size_t inputDimensions,
                                      unsigned int originalInputRank,
                                      const std::vector<unsigned int>& armnnAxes,
                                      arm_compute::Coordinates& outAclCoords)
{
    if (armnnAxes.empty())
    {
        // If no reduction axes were provided, then the input must be reduced along all dimensions.
        // Since arm_compute::CLReduceMean does not accept an empty vector as the reduction dimensions, we then
        // manually create a vector including all the input dimensions (in reversed order) as:
        //
        // { inputDimensions - 1, inputDimensions - 2, ..., 1, 0 }
        //
        outAclCoords.set_num_dimensions(inputDimensions);
        std::generate(outAclCoords.begin(), outAclCoords.end(), [d = inputDimensions - 1] () mutable { return d--; });
    }
    else
    {
        // Create a vector of reduction dimensions (in reversed order) with the given reduction axes.
        //
        // Adjust the given reduction axes according to the original rank of the input tensor (before ACL applied any
        // dimension correction).
        // For example, if the input tensor originally had 4 dimensions, and one of the reduction axes was 2, then the
        // new value for that reduction axis should be 1.
        //
        // Example:
        // ArmNN input shape = { 1, 1, 3, 2 } -> ACL input shape = { 2, 3 }
        // ArmNN reduction axis = { 2 }       -> ACL reduction axis = { 1 }
        // ArmNN reduction axis = { 3 }       -> ACL reduction axis = { 0 }
        //
        // The transformation: ACL reduction axis index = original rank - ArmNN reduction axis index - 1
        //
        outAclCoords.set_num_dimensions(armnnAxes.size());
        std::transform(armnnAxes.begin(), armnnAxes.end(),
                       outAclCoords.begin(),
                       [originalInputRank](unsigned int i){ return originalInputRank - i - 1; });
    }
}

} // anonymous namespace

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClMeanValidate(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const MeanDescriptor& desc)
{
    const arm_compute::TensorInfo aclInputInfo  = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    arm_compute::Coordinates coords;
    ConvertArmnnAxesToAclCoordinates(aclInputInfo.num_dimensions(),
                                     input.GetNumDimensions(),
                                     desc.m_Axis,
                                     coords);

    return arm_compute::CLReduceMean::validate(&aclInputInfo, coords, desc.m_KeepDims, &aclOutputInfo);
}

ClMeanWorkload::ClMeanWorkload(const MeanQueueDescriptor& descriptor, const WorkloadInfo& info)
    : BaseWorkload<MeanQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClMeanWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::Coordinates coords;
    ConvertArmnnAxesToAclCoordinates(input.info()->num_dimensions(),
                                     info.m_InputTensorInfos[0].GetNumDimensions(),
                                     m_Data.m_Parameters.m_Axis,
                                     coords);

    m_Layer.configure(&input, coords, m_Data.m_Parameters.m_KeepDims, &output);
}

void ClMeanWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClMeanWorkload_Execute");
    m_Layer.run();
}

} //namespace armnn
