//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClTileWorkload.hpp"
#include "ClWorkloadUtils.hpp"
#include <aclCommon/ArmComputeUtils.hpp>
#include <cl/ClTensorHandle.hpp>
#include <vector>
#include <algorithm>

using namespace armnn::armcomputetensorutils;
namespace armnn
{
arm_compute::Status ClTileWorkloadValidate(const TensorInfo& input,
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

    std::vector<uint32_t> aclMultiples = descriptor.m_Multiples;
    std::reverse(aclMultiples.begin(),aclMultiples.end());

    return arm_compute::CLTile::validate(&aclInput, &aclOutput, aclMultiples);
}

ClTileWorkload::ClTileWorkload(const armnn::TileQueueDescriptor& descriptor,
                               const armnn::WorkloadInfo& info,
                               const arm_compute::CLCompileContext& clCompileContext)
    : BaseWorkload<TileQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClTileWorkload", 1, 1);

    std::vector<uint32_t> aclMultiples = descriptor.m_Parameters.m_Multiples;
    std::reverse(aclMultiples.begin(),aclMultiples.end());

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    m_Layer.configure(clCompileContext, &input, &output, aclMultiples);
}

void ClTileWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClTileWorkload_Execute", this->GetGuid());
    m_Layer.run();
}

} //namespace armnn