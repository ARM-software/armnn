//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "NeonTileWorkload.hpp"
#include "NeonWorkloadUtils.hpp"
#include <aclCommon/ArmComputeUtils.hpp>
#include <vector>
#include <algorithm>

using namespace armnn::armcomputetensorutils;
namespace armnn
{
arm_compute::Status NeonTileWorkloadValidate(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const TileDescriptor& descriptor)
{
    if(input.GetDataType() == DataType::Boolean)
    {
        return arm_compute::Status{arm_compute::ErrorCode::RUNTIME_ERROR,
                                    "NeonTileWorkloadValidate: Unsupported Boolean DataType"};
    }

    const arm_compute::TensorInfo aclInput = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = BuildArmComputeTensorInfo(output);

    std::vector<unsigned int> aclMultiples = descriptor.m_Multiples;
    std::reverse(aclMultiples.begin(),aclMultiples.end());

    return arm_compute::NETile::validate(&aclInput, &aclOutput, aclMultiples);
}

NeonTileWorkload::NeonTileWorkload(const armnn::TileQueueDescriptor& descriptor,
                                   const armnn::WorkloadInfo& info)
        : BaseWorkload<TileQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonTileWorkload", 1, 1);

    std::vector<unsigned int> aclMultiples = descriptor.m_Parameters.m_Multiples;
    std::reverse(aclMultiples.begin(),aclMultiples.end());

    arm_compute::ITensor& input = static_cast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = static_cast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    m_Layer.configure(&input, &output, aclMultiples);
}

void NeonTileWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_NAME_GUID("NeonTileWorkload_Execute");
    m_Layer.run();
}
} //namespace armnn