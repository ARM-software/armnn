//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClAdditionFloat32Workload.hpp"

#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "backends/ArmComputeTensorUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

ClAdditionFloat32Workload::ClAdditionFloat32Workload(const AdditionQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info)
    : Float32Workload<AdditionQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClAdditionFloat32Workload", 2, 1);

    arm_compute::ICLTensor& input0 = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& input1 = static_cast<IClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    m_Layer.configure(&input0, &input1, &output, ms_AclConvertPolicy);
}

void ClAdditionFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "ClAdditionFloat32Workload_Execute");
    m_Layer.run();
}

bool ClAdditionFloat32Workload::IsSupported(const TensorInfo& input0,
                                            const TensorInfo& input1,
                                            const TensorInfo& output,
                                            std::string* reasonIfUnsupported)
{
    const arm_compute::TensorInfo aclInput0Info = BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput1Info = BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    const arm_compute::Status aclStatus = decltype(m_Layer)::validate(&aclInput0Info,
                                                                      &aclInput1Info,
                                                                      &aclOutputInfo,
                                                                      ms_AclConvertPolicy);

    const bool supported = (aclStatus.error_code() == arm_compute::ErrorCode::OK);
    if (!supported && reasonIfUnsupported)
    {
        *reasonIfUnsupported = aclStatus.error_description();
    }

    return supported;
}

} //namespace armnn