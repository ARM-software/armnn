//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonSpaceToDepthWorkload.hpp"
#include "NeonWorkloadUtils.hpp"

#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <ResolveType.hpp>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status NeonSpaceToDepthWorkloadValidate(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const SpaceToDepthDescriptor& descriptor)
{
    DataLayout dataLayout = descriptor.m_DataLayout;
    const arm_compute::TensorInfo aclInput = BuildArmComputeTensorInfo(input, dataLayout);
    const arm_compute::TensorInfo aclOutput = BuildArmComputeTensorInfo(output, dataLayout);

    int32_t blockSize  = armnn::numeric_cast<int32_t>(descriptor.m_BlockSize);

    return arm_compute::NESpaceToDepthLayer::validate(&aclInput, &aclOutput, blockSize);
}

NeonSpaceToDepthWorkload::NeonSpaceToDepthWorkload(const SpaceToDepthQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info)
    : NeonBaseWorkload<SpaceToDepthQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonSpaceToDepthWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("NeonSpaceToDepthWorkload", 1, 1);

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    input.info()->set_data_layout(aclDataLayout);

    int32_t blockSize = armnn::numeric_cast<int32_t>(descriptor.m_Parameters.m_BlockSize);

    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    output.info()->set_data_layout(aclDataLayout);

    m_Layer.reset(new arm_compute::NESpaceToDepthLayer());
    m_Layer->configure(&input, &output, blockSize);
    m_Layer->prepare();
}

void NeonSpaceToDepthWorkload::Execute() const
{
    if (m_Layer)
    {
        ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonSpaceToDepthWorkload_Execute", this->GetGuid());
        m_Layer->run();
    }
}

} //namespace armnn