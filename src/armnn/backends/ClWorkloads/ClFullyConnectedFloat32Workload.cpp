//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClFullyConnectedFloat32Workload.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "backends/ArmComputeTensorUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

ClFullyConnectedFloat32Workload::ClFullyConnectedFloat32Workload(const FullyConnectedQueueDescriptor& descriptor,
                                                                 const WorkloadInfo& info)
    : Float32Workload<FullyConnectedQueueDescriptor>(descriptor, info)
{

    BuildArmComputeTensor(m_WeightsTensor, m_Data.m_Weight->GetTensorInfo());

    arm_compute::CLTensor* optionalBiasTensor = nullptr;
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        BuildArmComputeTensor(m_BiasesTensor, m_Data.m_Bias->GetTensorInfo());
        optionalBiasTensor = &m_BiasesTensor;
    }

    m_Data.ValidateInputsOutputs("ClFullyConnectedFloat32Workload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    // Construct
    m_FullyConnected.configure(
        &input, &m_WeightsTensor, optionalBiasTensor, &output, m_Data.m_Parameters.m_TransposeWeightMatrix);

    // Allocate
    InitialiseArmComputeClTensorData(m_WeightsTensor, m_Data.m_Weight->GetConstTensor<float>());

    if (optionalBiasTensor)
    {
        InitialiseArmComputeClTensorData(*optionalBiasTensor, m_Data.m_Bias->GetConstTensor<float>());
    }
}

void ClFullyConnectedFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "ClFullyConnectedFloat32Workload_Execute");
    m_FullyConnected.run();
}

} //namespace armnn