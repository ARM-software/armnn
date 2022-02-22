//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonLstmFloatWorkload.hpp"
#include "NeonWorkloadUtils.hpp"

#include "aclCommon/ArmComputeTensorUtils.hpp"

#include <armnn/utility/NumericCast.hpp>

#include "neon/NeonTensorHandle.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

NeonLstmFloatWorkload::NeonLstmFloatWorkload(const LstmQueueDescriptor &descriptor, const WorkloadInfo &info)
        : FloatWorkload<LstmQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonLstmFloatWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    arm_compute::LSTMParams<arm_compute::ITensor> lstm_param;

    // Basic parameters
    m_InputToForgetWeightsTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_InputToForgetWeightsTensor, m_Data.m_InputToForgetWeights->GetTensorInfo());

    m_InputToCellWeightsTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_InputToCellWeightsTensor, m_Data.m_InputToCellWeights->GetTensorInfo());

    m_InputToOutputWeightsTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_InputToOutputWeightsTensor, m_Data.m_InputToOutputWeights->GetTensorInfo());

    m_RecurrentToForgetWeightsTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_RecurrentToForgetWeightsTensor, m_Data.m_RecurrentToForgetWeights->GetTensorInfo());

    m_RecurrentToCellWeightsTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_RecurrentToCellWeightsTensor, m_Data.m_RecurrentToCellWeights->GetTensorInfo());

    m_RecurrentToOutputWeightsTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_RecurrentToOutputWeightsTensor, m_Data.m_RecurrentToOutputWeights->GetTensorInfo());

    m_ForgetGateBiasTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_ForgetGateBiasTensor, m_Data.m_ForgetGateBias->GetTensorInfo());

    m_CellBiasTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_CellBiasTensor, m_Data.m_CellBias->GetTensorInfo());

    m_OutputGateBiasTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_OutputGateBiasTensor, m_Data.m_OutputGateBias->GetTensorInfo());

    // for future reference: check the AndroidNN API for the logic here
    if (!m_Data.m_Parameters.m_CifgEnabled)
    {
        m_InputToInputWeightsTensor = std::make_unique<arm_compute::Tensor>();
        BuildArmComputeTensor(*m_InputToInputWeightsTensor, m_Data.m_InputToInputWeights->GetTensorInfo());

        m_RecurrentToInputWeightsTensor = std::make_unique<arm_compute::Tensor>();
        BuildArmComputeTensor(*m_RecurrentToInputWeightsTensor, m_Data.m_RecurrentToInputWeights->GetTensorInfo());

        m_CellToInputWeightsTensor = std::make_unique<arm_compute::Tensor>();
        if (m_Data.m_CellToInputWeights != nullptr)
        {
            BuildArmComputeTensor(*m_CellToInputWeightsTensor, m_Data.m_CellToInputWeights->GetTensorInfo());
        }

        m_InputGateBiasTensor = std::make_unique<arm_compute::Tensor>();
        BuildArmComputeTensor(*m_InputGateBiasTensor, m_Data.m_InputGateBias->GetTensorInfo());

        lstm_param.set_cifg_params(m_InputToInputWeightsTensor.get(),
                                   m_RecurrentToInputWeightsTensor.get(),
                                   m_Data.m_CellToInputWeights != nullptr ? m_CellToInputWeightsTensor.get() : nullptr,
                                   m_InputGateBiasTensor.get());
    }

    if (m_Data.m_Parameters.m_ProjectionEnabled)
    {
        m_ProjectionWeightsTensor = std::make_unique<arm_compute::Tensor>();
        BuildArmComputeTensor(*m_ProjectionWeightsTensor, m_Data.m_ProjectionWeights->GetTensorInfo());

        m_ProjectionBiasTensor = std::make_unique<arm_compute::Tensor>();
        if (m_Data.m_ProjectionBias != nullptr)
        {
            BuildArmComputeTensor(*m_ProjectionBiasTensor, m_Data.m_ProjectionBias->GetTensorInfo());
        }

        lstm_param.set_projection_params(m_ProjectionWeightsTensor.get(),
                                         m_Data.m_ProjectionBias != nullptr ? m_ProjectionBiasTensor.get() : nullptr);
    }

    if (m_Data.m_Parameters.m_PeepholeEnabled)
    {
        m_CellToForgetWeightsTensor = std::make_unique<arm_compute::Tensor>();
        BuildArmComputeTensor(*m_CellToForgetWeightsTensor, m_Data.m_CellToForgetWeights->GetTensorInfo());

        m_CellToOutputWeightsTensor = std::make_unique<arm_compute::Tensor>();
        BuildArmComputeTensor(*m_CellToOutputWeightsTensor, m_Data.m_CellToOutputWeights->GetTensorInfo());

        lstm_param.set_peephole_params(m_CellToForgetWeightsTensor.get(), m_CellToOutputWeightsTensor.get());
    }

    if (m_Data.m_Parameters.m_LayerNormEnabled)
    {
        m_InputLayerNormWeightsTensor = std::make_unique<arm_compute::Tensor>();
        if (!m_Data.m_Parameters.m_CifgEnabled)
        {
            BuildArmComputeTensor(*m_InputLayerNormWeightsTensor, m_Data.m_InputLayerNormWeights->GetTensorInfo());
        }

        m_ForgetLayerNormWeightsTensor = std::make_unique<arm_compute::Tensor>();
        BuildArmComputeTensor(*m_ForgetLayerNormWeightsTensor, m_Data.m_ForgetLayerNormWeights->GetTensorInfo());

        m_CellLayerNormWeightsTensor = std::make_unique<arm_compute::Tensor>();
        BuildArmComputeTensor(*m_CellLayerNormWeightsTensor, m_Data.m_CellLayerNormWeights->GetTensorInfo());

        m_OutputLayerNormWeightsTensor = std::make_unique<arm_compute::Tensor>();
        BuildArmComputeTensor(*m_OutputLayerNormWeightsTensor, m_Data.m_OutputLayerNormWeights->GetTensorInfo());

        lstm_param.set_layer_normalization_params(m_Data.m_Parameters.m_CifgEnabled ?
                                                  nullptr : m_InputLayerNormWeightsTensor.get(),
                                                  m_ForgetLayerNormWeightsTensor.get(),
                                                  m_CellLayerNormWeightsTensor.get(),
                                                  m_OutputLayerNormWeightsTensor.get());
    }

    const arm_compute::ITensor& input           = static_cast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    const arm_compute::ITensor& output_state_in = static_cast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    const arm_compute::ITensor& cell_state_in   = static_cast<IAclTensorHandle*>(m_Data.m_Inputs[2])->GetTensor();

    arm_compute::ITensor& output_state_out      = static_cast<IAclTensorHandle*>(m_Data.m_Outputs[1])->GetTensor();
    arm_compute::ITensor& cell_state_out        = static_cast<IAclTensorHandle*>(m_Data.m_Outputs[2])->GetTensor();
    arm_compute::ITensor& output                = static_cast<IAclTensorHandle*>(m_Data.m_Outputs[3])->GetTensor();

    // Get the batch_size and the num_units from the cellStateIn dimensions
    const TensorInfo& inputTensorInfo = info.m_InputTensorInfos[2];
    const unsigned int batch_size = armnn::numeric_cast<unsigned int>(inputTensorInfo.GetShape()[0]);
    const unsigned int num_units  = armnn::numeric_cast<unsigned int>(inputTensorInfo.GetShape()[1]);

    m_ScratchBuffer = std::make_unique<arm_compute::Tensor>();
    if (m_Data.m_Parameters.m_CifgEnabled)
    {
        // 2D tensor with dimensions [num_units * 3, batch_size] with CIFG
        armnn::TensorInfo scratchBuffer1({ batch_size, num_units * 3 }, DataType::Float32);
        BuildArmComputeTensor(*m_ScratchBuffer, scratchBuffer1);
    }
    else
    {
        // scratch_buffer [num_units * 4, batch_size] without CIFG
        armnn::TensorInfo scratchBuffer2({ batch_size, num_units * 4 }, DataType::Float32);
        BuildArmComputeTensor(*m_ScratchBuffer, scratchBuffer2);
    }

    float cell_threshold = m_Data.m_Parameters.m_ClippingThresCell;
    float projection_threshold = m_Data.m_Parameters.m_ClippingThresProj;

    // for preparing the object for the class ActivationLayerInfo, we need to consider 5 situations
    arm_compute::ActivationLayerInfo activationLayerInfo;
    if (m_Data.m_Parameters.m_ActivationFunc == 0)
    {
        // no activation, do nothing
    }
    else if (m_Data.m_Parameters.m_ActivationFunc == 1)
    {
        activationLayerInfo = arm_compute::ActivationLayerInfo(
                arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
    }
    else if (m_Data.m_Parameters.m_ActivationFunc == 3)
    {
        activationLayerInfo = arm_compute::ActivationLayerInfo(
                arm_compute::ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.0);
    }
    else if (m_Data.m_Parameters.m_ActivationFunc == 4)
    {
        activationLayerInfo = arm_compute::ActivationLayerInfo(
                arm_compute::ActivationLayerInfo::ActivationFunction::TANH, 1.0, 1.0);
    }
    else if (m_Data.m_Parameters.m_ActivationFunc == 6)
    {
        activationLayerInfo = arm_compute::ActivationLayerInfo(
                arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC);
    }
    else
    {
        throw armnn::Exception("Wrong Type of Activation Function!");
    }


    m_LstmLayer.configure(&input, m_InputToForgetWeightsTensor.get(), m_InputToCellWeightsTensor.get(),
                          m_InputToOutputWeightsTensor.get(), m_RecurrentToForgetWeightsTensor.get(),
                          m_RecurrentToCellWeightsTensor.get(), m_RecurrentToOutputWeightsTensor.get(),
                          m_ForgetGateBiasTensor.get(), m_CellBiasTensor.get(), m_OutputGateBiasTensor.get(),
                          &output_state_in, &cell_state_in, m_ScratchBuffer.get(), &output_state_out,
                          &cell_state_out, &output, lstm_param, activationLayerInfo,
                          cell_threshold, projection_threshold);

    armcomputetensorutils::InitialiseArmComputeTensorEmpty(*m_ScratchBuffer);

    InitializeArmComputeTensorData(*m_InputToForgetWeightsTensor,
                                   m_Data.m_InputToForgetWeights);
    InitializeArmComputeTensorData(*m_InputToCellWeightsTensor,
                                   m_Data.m_InputToCellWeights);
    InitializeArmComputeTensorData(*m_InputToOutputWeightsTensor,
                                   m_Data.m_InputToOutputWeights);
    InitializeArmComputeTensorData(*m_RecurrentToForgetWeightsTensor,
                                   m_Data.m_RecurrentToForgetWeights);
    InitializeArmComputeTensorData(*m_RecurrentToCellWeightsTensor,
                                   m_Data.m_RecurrentToCellWeights);
    InitializeArmComputeTensorData(*m_RecurrentToOutputWeightsTensor,
                                   m_Data.m_RecurrentToOutputWeights);
    InitializeArmComputeTensorData(*m_ForgetGateBiasTensor,
                                   m_Data.m_ForgetGateBias);
    InitializeArmComputeTensorData(*m_CellBiasTensor,
                                   m_Data.m_CellBias);
    InitializeArmComputeTensorData(*m_OutputGateBiasTensor,
                                   m_Data.m_OutputGateBias);

    if (!m_Data.m_Parameters.m_CifgEnabled)
    {
        InitializeArmComputeTensorData(*m_InputToInputWeightsTensor,
                                       m_Data.m_InputToInputWeights);
        InitializeArmComputeTensorData(*m_RecurrentToInputWeightsTensor,
                                       m_Data.m_RecurrentToInputWeights);
        if (m_Data.m_CellToInputWeights != nullptr)
        {
            InitializeArmComputeTensorData(*m_CellToInputWeightsTensor,
                                           m_Data.m_CellToInputWeights);
        }
        InitializeArmComputeTensorData(*m_InputGateBiasTensor,
                                       m_Data.m_InputGateBias);
    }

    if (m_Data.m_Parameters.m_ProjectionEnabled)
    {
        InitializeArmComputeTensorData(*m_ProjectionWeightsTensor,
                                       m_Data.m_ProjectionWeights);
        if (m_Data.m_ProjectionBias != nullptr)
        {
            InitializeArmComputeTensorData(*m_ProjectionBiasTensor,
                                           m_Data.m_ProjectionBias);
        }
    }

    if (m_Data.m_Parameters.m_PeepholeEnabled)
    {
        InitializeArmComputeTensorData(*m_CellToForgetWeightsTensor,
                                       m_Data.m_CellToForgetWeights);
        InitializeArmComputeTensorData(*m_CellToOutputWeightsTensor,
                                       m_Data.m_CellToOutputWeights);
    }

    if (m_Data.m_Parameters.m_LayerNormEnabled)
    {
        if (!m_Data.m_Parameters.m_CifgEnabled)
        {
            InitializeArmComputeTensorData(*m_InputLayerNormWeightsTensor, m_Data.m_InputLayerNormWeights);
        }
        InitializeArmComputeTensorData(*m_ForgetLayerNormWeightsTensor, m_Data.m_ForgetLayerNormWeights);
        InitializeArmComputeTensorData(*m_CellLayerNormWeightsTensor, m_Data.m_CellLayerNormWeights);
        InitializeArmComputeTensorData(*m_OutputLayerNormWeightsTensor, m_Data.m_OutputLayerNormWeights);
    }

    // Force Compute Library to perform the necessary copying and reshaping, after which
    // delete all the input tensors that will no longer be needed
    m_LstmLayer.prepare();
    FreeUnusedTensors();
}

void NeonLstmFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonLstmFloatWorkload_Execute", this->GetGuid());
    m_LstmLayer.run();
}

arm_compute::Status NeonLstmFloatWorkloadValidate(const TensorInfo& input,
                                                  const TensorInfo& outputStateIn,
                                                  const TensorInfo& cellStateIn,
                                                  const TensorInfo& scratchBuffer,
                                                  const TensorInfo& outputStateOut,
                                                  const TensorInfo& cellStateOut,
                                                  const TensorInfo& output,
                                                  const LstmDescriptor& descriptor,
                                                  const LstmInputParamsInfo& paramsInfo)
{
    arm_compute::LSTMParams<arm_compute::ITensorInfo> lstm_params_info;

    // The inputs and outputs
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputStateInInfo = BuildArmComputeTensorInfo(outputStateIn);
    const arm_compute::TensorInfo aclCellStateInInfo = BuildArmComputeTensorInfo(cellStateIn);
    const arm_compute::TensorInfo aclScratchBufferInfo = BuildArmComputeTensorInfo(scratchBuffer);
    const arm_compute::TensorInfo aclOutputStateOutInfo = BuildArmComputeTensorInfo(outputStateOut);
    const arm_compute::TensorInfo aclCellStateOutInfo = BuildArmComputeTensorInfo(cellStateOut);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    // Basic parameters
    const arm_compute::TensorInfo aclInputToForgetWeightsInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetInputToForgetWeights());
    const arm_compute::TensorInfo aclInputToCellWeightsInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetInputToCellWeights());
    const arm_compute::TensorInfo aclInputToOutputWeightsInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetInputToOutputWeights());
    const arm_compute::TensorInfo aclRecurrentToForgetWeightsInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetRecurrentToForgetWeights());
    const arm_compute::TensorInfo aclRecurrentToCellWeightsInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetRecurrentToCellWeights());
    const arm_compute::TensorInfo aclRecurrentToOutputWeightsInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetRecurrentToOutputWeights());
    const arm_compute::TensorInfo aclForgetGateBiasInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetForgetGateBias());
    const arm_compute::TensorInfo aclCellBiasInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetCellBias());
    const arm_compute::TensorInfo aclOutputGateBiasInfo
                                  = BuildArmComputeTensorInfo(paramsInfo.GetOutputGateBias());

    arm_compute::TensorInfo aclInputToInputWeightsInfo;
    arm_compute::TensorInfo aclRecurrentToInputWeightsInfo;
    arm_compute::TensorInfo aclCellToInputWeightsInfo;
    arm_compute::TensorInfo aclInputGateBiasInfo;
    arm_compute::TensorInfo aclProjectionWeightsInfo;
    arm_compute::TensorInfo aclProjectionBiasInfo;
    arm_compute::TensorInfo aclCellToForgetWeightsInfo;
    arm_compute::TensorInfo aclCellToOutputWeightsInfo;

    arm_compute::TensorInfo aclInputLayerNormWeightsInfo;
    arm_compute::TensorInfo aclForgetLayerNormWeightsInfo;
    arm_compute::TensorInfo aclCellLayerNormWeightsInfo;
    arm_compute::TensorInfo aclOutputLayerNormWeightsInfo;


    if (!descriptor.m_CifgEnabled)
    {
        if (descriptor.m_PeepholeEnabled)
        {
            aclCellToInputWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetCellToInputWeights());
        }
        aclInputToInputWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetInputToInputWeights());
        aclRecurrentToInputWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetRecurrentToInputWeights());
        aclInputGateBiasInfo = BuildArmComputeTensorInfo(paramsInfo.GetInputGateBias());

        lstm_params_info.set_cifg_params(&aclInputToInputWeightsInfo, &aclRecurrentToInputWeightsInfo,
                                         descriptor.m_PeepholeEnabled ? &aclCellToInputWeightsInfo : nullptr,
                                         &aclInputGateBiasInfo);
    }

    if (descriptor.m_ProjectionEnabled)
    {
        if (paramsInfo.m_ProjectionBias != nullptr)
        {
            aclProjectionBiasInfo = BuildArmComputeTensorInfo(paramsInfo.GetProjectionBias());
        }
        aclProjectionWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetProjectionWeights());

        lstm_params_info.set_projection_params(&aclProjectionWeightsInfo,
                                               paramsInfo.m_ProjectionBias != nullptr ?
                                               &aclProjectionBiasInfo : nullptr);
    }

    if (descriptor.m_PeepholeEnabled)
    {
        aclCellToForgetWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetCellToForgetWeights());
        aclCellToOutputWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetCellToOutputWeights());

        lstm_params_info.set_peephole_params(&aclCellToForgetWeightsInfo, &aclCellToOutputWeightsInfo);
    }

    if (descriptor.m_LayerNormEnabled)
    {
        if (!descriptor.m_CifgEnabled)
        {
            aclInputLayerNormWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetInputLayerNormWeights());
        }
        aclForgetLayerNormWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetForgetLayerNormWeights());
        aclCellLayerNormWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetCellLayerNormWeights());
        aclOutputLayerNormWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetOutputLayerNormWeights());

        lstm_params_info.set_layer_normalization_params(descriptor.m_CifgEnabled ?
                                                        nullptr : &aclInputLayerNormWeightsInfo,
                                                        &aclForgetLayerNormWeightsInfo,
                                                        &aclCellLayerNormWeightsInfo,
                                                        &aclOutputLayerNormWeightsInfo);
    }

    float cell_threshold = descriptor.m_ClippingThresCell;
    float projection_threshold = descriptor.m_ClippingThresProj;

    // for preparing the object for the class ActivationLayerInfo, we need to consider 5 situations
    arm_compute::ActivationLayerInfo activationLayerInfo;
    switch (descriptor.m_ActivationFunc)
    {
        case 0:
            // no activation, do nothing
            break;
        case 1:
            activationLayerInfo = arm_compute::ActivationLayerInfo(
                    arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
            break;
        case 3:
            activationLayerInfo = arm_compute::ActivationLayerInfo(
                    arm_compute::ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.0);
            break;
        case 4:
            activationLayerInfo = arm_compute::ActivationLayerInfo(
                    arm_compute::ActivationLayerInfo::ActivationFunction::TANH, 1.0, 1.0);
            break;
        case 6:
            activationLayerInfo = arm_compute::ActivationLayerInfo(
                    arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC);
            break;
        default:
            throw armnn::Exception("Wrong Type of Activation Function!");
    }

    return arm_compute::NELSTMLayer::validate(&aclInputInfo,
                                              &aclInputToForgetWeightsInfo,
                                              &aclInputToCellWeightsInfo,
                                              &aclInputToOutputWeightsInfo,
                                              &aclRecurrentToForgetWeightsInfo,
                                              &aclRecurrentToCellWeightsInfo,
                                              &aclRecurrentToOutputWeightsInfo,
                                              &aclForgetGateBiasInfo,
                                              &aclCellBiasInfo,
                                              &aclOutputGateBiasInfo,
                                              &aclOutputStateInInfo,
                                              &aclCellStateInInfo,
                                              &aclScratchBufferInfo,
                                              &aclOutputStateOutInfo,
                                              &aclCellStateOutInfo,
                                              &aclOutputInfo,
                                              lstm_params_info,
                                              activationLayerInfo,
                                              cell_threshold,
                                              projection_threshold);
}

void NeonLstmFloatWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_InputToInputWeightsTensor);
    FreeTensorIfUnused(m_InputToForgetWeightsTensor);
    FreeTensorIfUnused(m_InputToCellWeightsTensor);
    FreeTensorIfUnused(m_InputToOutputWeightsTensor);
    FreeTensorIfUnused(m_RecurrentToInputWeightsTensor);
    FreeTensorIfUnused(m_RecurrentToForgetWeightsTensor);
    FreeTensorIfUnused(m_RecurrentToCellWeightsTensor);
    FreeTensorIfUnused(m_RecurrentToOutputWeightsTensor);
    FreeTensorIfUnused(m_CellToInputWeightsTensor);
    FreeTensorIfUnused(m_CellToForgetWeightsTensor);
    FreeTensorIfUnused(m_CellToOutputWeightsTensor);
    FreeTensorIfUnused(m_InputGateBiasTensor);
    FreeTensorIfUnused(m_ForgetGateBiasTensor);
    FreeTensorIfUnused(m_CellBiasTensor);
    FreeTensorIfUnused(m_OutputGateBiasTensor);
    FreeTensorIfUnused(m_ProjectionWeightsTensor);
    FreeTensorIfUnused(m_ProjectionBiasTensor);
    FreeTensorIfUnused(m_ScratchBuffer);
    FreeTensorIfUnused(m_InputLayerNormWeightsTensor);
    FreeTensorIfUnused(m_ForgetLayerNormWeightsTensor);
    FreeTensorIfUnused(m_CellLayerNormWeightsTensor);
    FreeTensorIfUnused(m_OutputLayerNormWeightsTensor);
}

void NeonLstmFloatWorkload::ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot)
{
    ITensorHandle* backupHandle = this->m_Data.m_Inputs[slot];
    this->m_Data.m_Inputs[slot] = tensorHandle;
    try
    {
        Reconfigure();
    }
    catch(armnn::UnimplementedException& e)
    {
        // Cannot reconfigure, revert the slot back and throw the exception.
        this->m_Data.m_Inputs[slot] = backupHandle;
        throw e;
    }
}

// Replace output tensor handle with the given TensorHandle
void NeonLstmFloatWorkload::ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot)
{
    ITensorHandle* backupHandle = this->m_Data.m_Inputs[slot];
    this->m_Data.m_Inputs[slot] = tensorHandle;
    try
    {
        Reconfigure();
    }
    catch(armnn::UnimplementedException& e)
    {
        // Cannot reconfigure, revert the slot back and throw the exception.
        this->m_Data.m_Inputs[slot] = backupHandle;
        throw e;
    }
}

void NeonLstmFloatWorkload::Reconfigure()
{
    throw armnn::UnimplementedException("Reconfigure not implemented for this workload");
}

} //namespace armnn
