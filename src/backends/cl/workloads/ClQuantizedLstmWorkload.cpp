//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClQuantizedLstmWorkload.hpp"
#include "ClWorkloadUtils.hpp"

#include <armnn/backends/TensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <cl/ClTensorHandle.hpp>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status ClQuantizedLstmWorkloadValidate(const TensorInfo& input, const TensorInfo& previousCellStateIn,
                                                    const TensorInfo& previousOutputIn, const TensorInfo& cellStateOut,
                                                    const TensorInfo& output,
                                                    const QuantizedLstmInputParamsInfo& paramsInfo)
{
    // Inputs
    const arm_compute::TensorInfo aclInputInfo               = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclPreviousCellStateInInfo = BuildArmComputeTensorInfo(previousCellStateIn);
    const arm_compute::TensorInfo aclPreviousOutputInInfo    = BuildArmComputeTensorInfo(previousOutputIn);

    // Outputs
    const arm_compute::TensorInfo aclCellStateOutInfo        = BuildArmComputeTensorInfo(cellStateOut);
    const arm_compute::TensorInfo aclOutputInfo              = BuildArmComputeTensorInfo(output);

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
    const arm_compute::TensorInfo aclInputGateBiasInfo  = BuildArmComputeTensorInfo(paramsInfo.GetInputGateBias());
    const arm_compute::TensorInfo aclForgetGateBiasInfo = BuildArmComputeTensorInfo(paramsInfo.GetForgetGateBias());
    const arm_compute::TensorInfo aclCellBiasInfo       = BuildArmComputeTensorInfo(paramsInfo.GetCellBias());
    const arm_compute::TensorInfo aclOutputGateBiasInfo = BuildArmComputeTensorInfo(paramsInfo.GetOutputGateBias());

    return arm_compute::CLLSTMLayerQuantized::validate(&aclInputInfo, &aclInputToInputWeightsInfo,
                                                       &aclInputToForgetWeightsInfo, &aclInputToCellWeightsInfo,
                                                       &aclInputToOutputWeightsInfo, &aclRecurrentToInputWeightsInfo,
                                                       &aclRecurrentToForgetWeightsInfo, &aclRecurrentToCellWeightsInfo,
                                                       &aclRecurrentToOutputWeightsInfo, &aclInputGateBiasInfo,
                                                       &aclForgetGateBiasInfo, &aclCellBiasInfo, &aclOutputGateBiasInfo,
                                                       &aclPreviousCellStateInInfo, &aclPreviousOutputInInfo,
                                                       &aclCellStateOutInfo, &aclOutputInfo);
}

ClQuantizedLstmWorkload::ClQuantizedLstmWorkload(const QuantizedLstmQueueDescriptor &descriptor,
                                                 const WorkloadInfo &info,
                                                 const arm_compute::CLCompileContext& clCompileContext)
                                                 : ClBaseWorkload<QuantizedLstmQueueDescriptor>(descriptor, info)
{
    m_InputToInputWeightsTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_InputToInputWeightsTensor, m_Data.m_InputToInputWeights->GetTensorInfo());

    m_InputToForgetWeightsTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_InputToForgetWeightsTensor, m_Data.m_InputToForgetWeights->GetTensorInfo());

    m_InputToCellWeightsTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_InputToCellWeightsTensor, m_Data.m_InputToCellWeights->GetTensorInfo());

    m_InputToOutputWeightsTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_InputToOutputWeightsTensor, m_Data.m_InputToOutputWeights->GetTensorInfo());

    m_RecurrentToInputWeightsTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_RecurrentToInputWeightsTensor, m_Data.m_RecurrentToInputWeights->GetTensorInfo());

    m_RecurrentToForgetWeightsTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_RecurrentToForgetWeightsTensor, m_Data.m_RecurrentToForgetWeights->GetTensorInfo());

    m_RecurrentToCellWeightsTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_RecurrentToCellWeightsTensor, m_Data.m_RecurrentToCellWeights->GetTensorInfo());

    m_RecurrentToOutputWeightsTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_RecurrentToOutputWeightsTensor, m_Data.m_RecurrentToOutputWeights->GetTensorInfo());

    m_InputGateBiasTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_InputGateBiasTensor, m_Data.m_InputGateBias->GetTensorInfo());

    m_ForgetGateBiasTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_ForgetGateBiasTensor, m_Data.m_ForgetGateBias->GetTensorInfo());

    m_CellBiasTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_CellBiasTensor, m_Data.m_CellBias->GetTensorInfo());

    m_OutputGateBiasTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_OutputGateBiasTensor, m_Data.m_OutputGateBias->GetTensorInfo());

    const arm_compute::ICLTensor& inputTensor         = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& cellStateInTensor         = static_cast<IClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    const arm_compute::ICLTensor& outputStateInTensor = static_cast<IClTensorHandle*>(m_Data.m_Inputs[2])->GetTensor();

    arm_compute::ICLTensor& cellStateOutTensor        = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    arm_compute::ICLTensor& outputStateOutTensor      = static_cast<IClTensorHandle*>(m_Data.m_Outputs[1])->GetTensor();

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClQuantizedLstmWorkload_configure");
        m_QuantizedLstmLayer.configure(clCompileContext, &inputTensor, m_InputToInputWeightsTensor.get(),
                                       m_InputToForgetWeightsTensor.get(),
                                       m_InputToCellWeightsTensor.get(), m_InputToOutputWeightsTensor.get(),
                                       m_RecurrentToInputWeightsTensor.get(), m_RecurrentToForgetWeightsTensor.get(),
                                       m_RecurrentToCellWeightsTensor.get(), m_RecurrentToOutputWeightsTensor.get(),
                                       m_InputGateBiasTensor.get(), m_ForgetGateBiasTensor.get(),
                                       m_CellBiasTensor.get(),
                                       m_OutputGateBiasTensor.get(), &cellStateInTensor, &outputStateInTensor,
                                       &cellStateOutTensor, &outputStateOutTensor);
    }

    InitializeArmComputeClTensorData(*m_InputToInputWeightsTensor,      m_Data.m_InputToInputWeights);
    InitializeArmComputeClTensorData(*m_InputToForgetWeightsTensor,     m_Data.m_InputToForgetWeights);
    InitializeArmComputeClTensorData(*m_InputToCellWeightsTensor,       m_Data.m_InputToCellWeights);
    InitializeArmComputeClTensorData(*m_InputToOutputWeightsTensor,     m_Data.m_InputToOutputWeights);
    InitializeArmComputeClTensorData(*m_RecurrentToInputWeightsTensor,  m_Data.m_RecurrentToInputWeights);
    InitializeArmComputeClTensorData(*m_RecurrentToForgetWeightsTensor, m_Data.m_RecurrentToForgetWeights);
    InitializeArmComputeClTensorData(*m_RecurrentToCellWeightsTensor,   m_Data.m_RecurrentToCellWeights);
    InitializeArmComputeClTensorData(*m_RecurrentToOutputWeightsTensor, m_Data.m_RecurrentToOutputWeights);
    InitializeArmComputeClTensorData(*m_InputGateBiasTensor,            m_Data.m_InputGateBias);
    InitializeArmComputeClTensorData(*m_ForgetGateBiasTensor,           m_Data.m_ForgetGateBias);
    InitializeArmComputeClTensorData(*m_CellBiasTensor,                 m_Data.m_CellBias);
    InitializeArmComputeClTensorData(*m_OutputGateBiasTensor,           m_Data.m_OutputGateBias);

    m_QuantizedLstmLayer.prepare();
    FreeUnusedTensors();
}

void ClQuantizedLstmWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClQuantizedLstmWorkload_Execute", this->GetGuid());
    RunClFunction(m_QuantizedLstmLayer, CHECK_LOCATION());
}

void ClQuantizedLstmWorkload::FreeUnusedTensors()
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
}

} // namespace armnn