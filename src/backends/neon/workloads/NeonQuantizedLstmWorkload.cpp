//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonQuantizedLstmWorkload.hpp"
#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <neon/NeonTensorHandle.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

NeonQuantizedLstmWorkload::NeonQuantizedLstmWorkload(const QuantizedLstmQueueDescriptor &descriptor,
                                                     const WorkloadInfo &info)
        : NeonBaseWorkload<QuantizedLstmQueueDescriptor>(descriptor, info)
{
    // Basic parameters
    m_InputToInputWeightsTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_InputToInputWeightsTensor, m_Data.m_InputToInputWeights->GetTensorInfo());

    m_InputToForgetWeightsTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_InputToForgetWeightsTensor, m_Data.m_InputToForgetWeights->GetTensorInfo());

    m_InputToCellWeightsTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_InputToCellWeightsTensor, m_Data.m_InputToCellWeights->GetTensorInfo());

    m_InputToOutputWeightsTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_InputToOutputWeightsTensor, m_Data.m_InputToOutputWeights->GetTensorInfo());

    m_RecurrentToInputWeightsTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_RecurrentToInputWeightsTensor, m_Data.m_RecurrentToInputWeights->GetTensorInfo());

    m_RecurrentToForgetWeightsTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_RecurrentToForgetWeightsTensor, m_Data.m_RecurrentToForgetWeights->GetTensorInfo());

    m_RecurrentToCellWeightsTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_RecurrentToCellWeightsTensor, m_Data.m_RecurrentToCellWeights->GetTensorInfo());

    m_RecurrentToOutputWeightsTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_RecurrentToOutputWeightsTensor, m_Data.m_RecurrentToOutputWeights->GetTensorInfo());

    m_InputGateBiasTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_InputGateBiasTensor, m_Data.m_InputGateBias->GetTensorInfo());

    m_ForgetGateBiasTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_ForgetGateBiasTensor, m_Data.m_ForgetGateBias->GetTensorInfo());

    m_CellBiasTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_CellBiasTensor, m_Data.m_CellBias->GetTensorInfo());

    m_OutputGateBiasTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_OutputGateBiasTensor, m_Data.m_OutputGateBias->GetTensorInfo());

    const arm_compute::ITensor& input           = static_cast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& cell_state_in         = static_cast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    const arm_compute::ITensor& output_state_in = static_cast<IAclTensorHandle*>(m_Data.m_Inputs[2])->GetTensor();

    arm_compute::ITensor& cell_state_out        = static_cast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    arm_compute::ITensor& output_state_out      = static_cast<IAclTensorHandle*>(m_Data.m_Outputs[1])->GetTensor();

    m_QuantizedLstmLayer.configure(&input,
                                   m_InputToInputWeightsTensor.get(),
                                   m_InputToForgetWeightsTensor.get(),
                                   m_InputToCellWeightsTensor.get(),
                                   m_InputToOutputWeightsTensor.get(),
                                   m_RecurrentToInputWeightsTensor.get(),
                                   m_RecurrentToForgetWeightsTensor.get(),
                                   m_RecurrentToCellWeightsTensor.get(),
                                   m_RecurrentToOutputWeightsTensor.get(),
                                   m_InputGateBiasTensor.get(),
                                   m_ForgetGateBiasTensor.get(),
                                   m_CellBiasTensor.get(),
                                   m_OutputGateBiasTensor.get(),
                                   &cell_state_in,
                                   &output_state_in,
                                   &cell_state_out,
                                   &output_state_out);

    InitializeArmComputeTensorData(*m_InputToInputWeightsTensor,
                                   m_Data.m_InputToInputWeights);

    InitializeArmComputeTensorData(*m_InputToForgetWeightsTensor,
                                   m_Data.m_InputToForgetWeights);

    InitializeArmComputeTensorData(*m_InputToCellWeightsTensor,
                                   m_Data.m_InputToCellWeights);

    InitializeArmComputeTensorData(*m_InputToOutputWeightsTensor,
                                   m_Data.m_InputToOutputWeights);

    InitializeArmComputeTensorData(*m_RecurrentToInputWeightsTensor,
                                   m_Data.m_RecurrentToInputWeights);

    InitializeArmComputeTensorData(*m_RecurrentToForgetWeightsTensor,
                                   m_Data.m_RecurrentToForgetWeights);

    InitializeArmComputeTensorData(*m_RecurrentToCellWeightsTensor,
                                   m_Data.m_RecurrentToCellWeights);

    InitializeArmComputeTensorData(*m_RecurrentToOutputWeightsTensor,
                                   m_Data.m_RecurrentToOutputWeights);

    InitializeArmComputeTensorData(*m_InputGateBiasTensor,
                                   m_Data.m_InputGateBias);

    InitializeArmComputeTensorData(*m_ForgetGateBiasTensor,
                                   m_Data.m_ForgetGateBias);

    InitializeArmComputeTensorData(*m_CellBiasTensor,
                                   m_Data.m_CellBias);

    InitializeArmComputeTensorData(*m_OutputGateBiasTensor,
                                   m_Data.m_OutputGateBias);

    // Force Compute Library to perform the necessary copying and reshaping, after which
    // delete all the input tensors that will no longer be needed
    m_QuantizedLstmLayer.prepare();
    FreeUnusedTensors();
}

void NeonQuantizedLstmWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonQuantizedLstmWorkload_Execute", this->GetGuid());
    m_QuantizedLstmLayer.run();
}

arm_compute::Status NeonQuantizedLstmWorkloadValidate(const TensorInfo& input,
                                                      const TensorInfo& cellStateIn,
                                                      const TensorInfo& outputStateIn,
                                                      const TensorInfo& cellStateOut,
                                                      const TensorInfo& outputStateOut,
                                                      const QuantizedLstmInputParamsInfo& paramsInfo)
{
    // The inputs and outputs
    const arm_compute::TensorInfo aclInputInfo          = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclCellStateInInfo    = BuildArmComputeTensorInfo(cellStateIn);
    const arm_compute::TensorInfo aclOutputStateInInfo  = BuildArmComputeTensorInfo(outputStateIn);
    const arm_compute::TensorInfo aclCellStateOutInfo   = BuildArmComputeTensorInfo(cellStateOut);
    const arm_compute::TensorInfo aclOutputStateOutInfo = BuildArmComputeTensorInfo(outputStateOut);

    // Basic parameters
    const arm_compute::TensorInfo aclInputToInputWeightsInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetInputToInputWeights());
    const arm_compute::TensorInfo aclInputToForgetWeightsInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetInputToForgetWeights());
    const arm_compute::TensorInfo aclInputToCellWeightsInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetInputToCellWeights());
    const arm_compute::TensorInfo aclInputToOutputWeightsInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetInputToOutputWeights());

    const arm_compute::TensorInfo aclRecurrentToInputWeightsInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetRecurrentToInputWeights());
    const arm_compute::TensorInfo aclRecurrentToForgetWeightsInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetRecurrentToForgetWeights());
    const arm_compute::TensorInfo aclRecurrentToCellWeightsInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetRecurrentToCellWeights());
    const arm_compute::TensorInfo aclRecurrentToOutputWeightsInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetRecurrentToOutputWeights());

    const arm_compute::TensorInfo aclInputGateBiasInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetInputGateBias());
    const arm_compute::TensorInfo aclForgetGateBiasInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetForgetGateBias());
    const arm_compute::TensorInfo aclCellBiasInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetCellBias());
    const arm_compute::TensorInfo aclOutputGateBiasInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetOutputGateBias());

    return arm_compute::NELSTMLayerQuantized::validate(&aclInputInfo,
                                                       &aclInputToInputWeightsInfo,
                                                       &aclInputToForgetWeightsInfo,
                                                       &aclInputToCellWeightsInfo,
                                                       &aclInputToOutputWeightsInfo,
                                                       &aclRecurrentToInputWeightsInfo,
                                                       &aclRecurrentToForgetWeightsInfo,
                                                       &aclRecurrentToCellWeightsInfo,
                                                       &aclRecurrentToOutputWeightsInfo,
                                                       &aclInputGateBiasInfo,
                                                       &aclForgetGateBiasInfo,
                                                       &aclCellBiasInfo,
                                                       &aclOutputGateBiasInfo,
                                                       &aclCellStateInInfo,
                                                       &aclOutputStateInInfo,
                                                       &aclCellStateOutInfo,
                                                       &aclOutputStateOutInfo);
}

void NeonQuantizedLstmWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_InputToInputWeightsTensor);
    FreeTensorIfUnused(m_InputToForgetWeightsTensor);
    FreeTensorIfUnused(m_InputToCellWeightsTensor);
    FreeTensorIfUnused(m_InputToOutputWeightsTensor);
    FreeTensorIfUnused(m_RecurrentToInputWeightsTensor);
    FreeTensorIfUnused(m_RecurrentToForgetWeightsTensor);
    FreeTensorIfUnused(m_RecurrentToCellWeightsTensor);
    FreeTensorIfUnused(m_RecurrentToOutputWeightsTensor);
    FreeTensorIfUnused(m_InputGateBiasTensor);
    FreeTensorIfUnused(m_ForgetGateBiasTensor);
    FreeTensorIfUnused(m_CellBiasTensor);
    FreeTensorIfUnused(m_OutputGateBiasTensor);
    FreeTensorIfUnused(m_CellStateInTensor);
    FreeTensorIfUnused(m_OutputStateInTensor);
    FreeTensorIfUnused(m_CellStateOutTensor);
}

} //namespace armnn
