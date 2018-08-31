//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClConvertFp16ToFp32Workload.hpp"
#include "backends/ClTensorHandle.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

static constexpr arm_compute::ConvertPolicy g_AclConvertPolicy = arm_compute::ConvertPolicy::SATURATE;

ClConvertFp16ToFp32Workload::ClConvertFp16ToFp32Workload(
    const ConvertFp16ToFp32QueueDescriptor& descriptor, const WorkloadInfo& info) :
    Float16ToFloat32Workload<ConvertFp16ToFp32QueueDescriptor>(descriptor, info)
{
    this->m_Data.ValidateInputsOutputs("ClConvertFp16ToFp32Workload", 1, 1);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(this->m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(this->m_Data.m_Outputs[0])->GetTensor();

    m_Layer.configure(&input, &output, g_AclConvertPolicy, 0);
}

void ClConvertFp16ToFp32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClConvertFp16ToFp32Workload_Execute");
    m_Layer.run();
}

arm_compute::Status ClConvertFp16ToFp32WorkloadValidate(const TensorInfo& input,
                                                        const TensorInfo& output,
                                                        std::string* reasonIfUnsupported)
{
    if (input.GetDataType() != DataType::Float16)
    {
        *reasonIfUnsupported = "Input should be Float16";
        return arm_compute::Status(arm_compute::ErrorCode::RUNTIME_ERROR, *reasonIfUnsupported);
    }
    if (output.GetDataType() != DataType::Float32)
    {
        *reasonIfUnsupported = "Output should be Float32";
        return arm_compute::Status(arm_compute::ErrorCode::RUNTIME_ERROR, *reasonIfUnsupported);
    }

    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    const arm_compute::Status aclStatus = arm_compute::CLDepthConvertLayer::validate(
        &aclInputInfo, &aclOutputInfo, g_AclConvertPolicy, 0);

    const bool supported = (aclStatus.error_code() == arm_compute::ErrorCode::OK);
    if (!supported && reasonIfUnsupported)
    {
        *reasonIfUnsupported = aclStatus.error_description();
    }

    return aclStatus;
}


} //namespace armnn
