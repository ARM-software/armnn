//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonDepthToSpaceWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status NeonDepthToSpaceWorkloadValidate(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const DepthToSpaceDescriptor& descriptor)
{
    DataLayout dataLayout = descriptor.m_DataLayout;
    const arm_compute::TensorInfo aclInput = BuildArmComputeTensorInfo(input, dataLayout);
    const arm_compute::TensorInfo aclOutput = BuildArmComputeTensorInfo(output, dataLayout);

    int32_t blockSize = armnn::numeric_cast<int32_t>(descriptor.m_BlockSize);

    return arm_compute::NEDepthToSpaceLayer::validate(&aclInput, &aclOutput, blockSize);
}

NeonDepthToSpaceWorkload::NeonDepthToSpaceWorkload(const DepthToSpaceQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info)
    : NeonBaseWorkload<DepthToSpaceQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonDepthToSpaceWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("NeonDepthToSpaceWorkload", 1, 1);

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);

    arm_compute::ITensor& input =
            PolymorphicPointerDowncast<IAclTensorHandle>(m_Data.m_Inputs[0])->GetTensor();
    input.info()->set_data_layout(aclDataLayout);

    int32_t blockSize = armnn::numeric_cast<int32_t>(descriptor.m_Parameters.m_BlockSize);

    arm_compute::ITensor& output =
            PolymorphicPointerDowncast<IAclTensorHandle>(m_Data.m_Outputs[0])->GetTensor();
    output.info()->set_data_layout(aclDataLayout);

    m_Layer.configure(&input, &output, blockSize);
    m_Layer.prepare();
}

void NeonDepthToSpaceWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonDepthToSpaceWorkload_Execute", this->GetGuid());
    m_Layer.run();
}

} // namespace armnn
