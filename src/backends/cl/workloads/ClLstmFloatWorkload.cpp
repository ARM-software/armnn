//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClLstmFloatWorkload.hpp"
#include <cl/ClTensorHandle.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <armnn/utility/NumericCast.hpp>

#include <arm_compute/runtime/CL/functions/CLLSTMLayer.h>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

ClLstmFloatWorkload::ClLstmFloatWorkload(const LstmQueueDescriptor& descriptor,
                                         const WorkloadInfo& info,
                                         const arm_compute::CLCompileContext& clCompileContext)
        : FloatWorkload<LstmQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClLstmFloatWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         GetGuid());

    arm_compute::LSTMParams<arm_compute::ICLTensor> lstm_param;

    // Basic parameters
    m_InputToForgetWeightsTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_InputToForgetWeightsTensor, m_Data.m_InputToForgetWeights->GetTensorInfo());

    m_InputToCellWeightsTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_InputToCellWeightsTensor, m_Data.m_InputToCellWeights->GetTensorInfo());

    m_InputToOutputWeightsTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_InputToOutputWeightsTensor, m_Data.m_InputToOutputWeights->GetTensorInfo());

    m_RecurrentToForgetWeightsTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_RecurrentToForgetWeightsTensor, m_Data.m_RecurrentToForgetWeights->GetTensorInfo());

    m_RecurrentToCellWeightsTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_RecurrentToCellWeightsTensor, m_Data.m_RecurrentToCellWeights->GetTensorInfo());

    m_RecurrentToOutputWeightsTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_RecurrentToOutputWeightsTensor, m_Data.m_RecurrentToOutputWeights->GetTensorInfo());

    m_ForgetGateBiasTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_ForgetGateBiasTensor, m_Data.m_ForgetGateBias->GetTensorInfo());

    m_CellBiasTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_CellBiasTensor, m_Data.m_CellBias->GetTensorInfo());

    m_OutputGateBiasTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_OutputGateBiasTensor, m_Data.m_OutputGateBias->GetTensorInfo());

    // for future reference: check the AndroidNN API for the logic here
    if (!m_Data.m_Parameters.m_CifgEnabled)
    {
        m_InputToInputWeightsTensor = std::make_unique<arm_compute::CLTensor>();
        BuildArmComputeTensor(*m_InputToInputWeightsTensor, m_Data.m_InputToInputWeights->GetTensorInfo());

        m_RecurrentToInputWeightsTensor = std::make_unique<arm_compute::CLTensor>();
        BuildArmComputeTensor(*m_RecurrentToInputWeightsTensor, m_Data.m_RecurrentToInputWeights->GetTensorInfo());

        m_CellToInputWeightsTensor = std::make_unique<arm_compute::CLTensor>();
        if (m_Data.m_CellToInputWeights != nullptr)
        {
            BuildArmComputeTensor(*m_CellToInputWeightsTensor, m_Data.m_CellToInputWeights->GetTensorInfo());
        }

        m_InputGateBiasTensor = std::make_unique<arm_compute::CLTensor>();
        BuildArmComputeTensor(*m_InputGateBiasTensor, m_Data.m_InputGateBias->GetTensorInfo());

        lstm_param.set_cifg_params(m_InputToInputWeightsTensor.get(),
                                   m_RecurrentToInputWeightsTensor.get(),
                                   m_Data.m_CellToInputWeights != nullptr ? m_CellToInputWeightsTensor.get() : nullptr,
                                   m_InputGateBiasTensor.get());
    }

    if (m_Data.m_Parameters.m_ProjectionEnabled)
    {
        m_ProjectionWeightsTensor = std::make_unique<arm_compute::CLTensor>();
        BuildArmComputeTensor(*m_ProjectionWeightsTensor, m_Data.m_ProjectionWeights->GetTensorInfo());

        m_ProjectionBiasTensor = std::make_unique<arm_compute::CLTensor>();
        if (m_Data.m_ProjectionBias != nullptr)
        {
            BuildArmComputeTensor(*m_ProjectionBiasTensor, m_Data.m_ProjectionBias->GetTensorInfo());
        }

        lstm_param.set_projection_params(m_ProjectionWeightsTensor.get(),
                                         m_Data.m_ProjectionBias != nullptr ? m_ProjectionBiasTensor.get() : nullptr);
    }

    if (m_Data.m_Parameters.m_PeepholeEnabled)
    {
        m_CellToForgetWeightsTensor = std::make_unique<arm_compute::CLTensor>();
        BuildArmComputeTensor(*m_CellToForgetWeightsTensor, m_Data.m_CellToForgetWeights->GetTensorInfo());

        m_CellToOutputWeightsTensor = std::make_unique<arm_compute::CLTensor>();
        BuildArmComputeTensor(*m_CellToOutputWeightsTensor, m_Data.m_CellToOutputWeights->GetTensorInfo());

        lstm_param.set_peephole_params(m_CellToForgetWeightsTensor.get(), m_CellToOutputWeightsTensor.get());
    }

    if (m_Data.m_Parameters.m_LayerNormEnabled)
    {
        m_InputLayerNormWeightsTensor  = std::make_unique<arm_compute::CLTensor>();
        m_ForgetLayerNormWeightsTensor = std::make_unique<arm_compute::CLTensor>();
        m_CellLayerNormWeightsTensor   = std::make_unique<arm_compute::CLTensor>();
        m_OutputLayerNormWeightsTensor = std::make_unique<arm_compute::CLTensor>();

        if (!m_Data.m_Parameters.m_CifgEnabled)
        {
            BuildArmComputeTensor(*m_InputLayerNormWeightsTensor, m_Data.m_InputLayerNormWeights->GetTensorInfo());
        }
        BuildArmComputeTensor(*m_ForgetLayerNormWeightsTensor, m_Data.m_ForgetLayerNormWeights->GetTensorInfo());
        BuildArmComputeTensor(*m_CellLayerNormWeightsTensor, m_Data.m_CellLayerNormWeights->GetTensorInfo());
        BuildArmComputeTensor(*m_OutputLayerNormWeightsTensor, m_Data.m_OutputLayerNormWeights->GetTensorInfo());

        lstm_param.set_layer_normalization_params(m_Data.m_Parameters.m_CifgEnabled ? nullptr :
                                                  m_InputLayerNormWeightsTensor.get(),
                                                  m_ForgetLayerNormWeightsTensor.get(),
                                                  m_CellLayerNormWeightsTensor.get(),
                                                  m_OutputLayerNormWeightsTensor.get());
    }

    const arm_compute::ICLTensor& input           = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    const arm_compute::ICLTensor& output_state_in = static_cast<IClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& cell_state_in         = static_cast<IClTensorHandle*>(m_Data.m_Inputs[2])->GetTensor();

    arm_compute::ICLTensor& output_state_out      = static_cast<IClTensorHandle*>(m_Data.m_Outputs[1])->GetTensor();
    arm_compute::ICLTensor& cell_state_out        = static_cast<IClTensorHandle*>(m_Data.m_Outputs[2])->GetTensor();
    arm_compute::ICLTensor& output                = static_cast<IClTensorHandle*>(m_Data.m_Outputs[3])->GetTensor();

    // Get the batch_size and the num_units from the cellStateIn dimensions
    const TensorInfo& inputTensorInfo = info.m_InputTensorInfos[2];
    const unsigned int batch_size = armnn::numeric_cast<unsigned int>(inputTensorInfo.GetShape()[0]);
    const unsigned int num_units  = armnn::numeric_cast<unsigned int>(inputTensorInfo.GetShape()[1]);

    m_ScratchBuffer = std::make_unique<arm_compute::CLTensor>();
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
    arm_compute::ActivationLayerInfo activationLayerInfo =
        ConvertLstmActivationFuncToAclLayerInfo(m_Data.m_Parameters.m_ActivationFunc);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClLstmFloatWorkload_configure");
        m_LstmLayer.configure(clCompileContext, &input, m_InputToForgetWeightsTensor.get(),
                              m_InputToCellWeightsTensor.get(), m_InputToOutputWeightsTensor.get(),
                              m_RecurrentToForgetWeightsTensor.get(), m_RecurrentToCellWeightsTensor.get(),
                              m_RecurrentToOutputWeightsTensor.get(), m_ForgetGateBiasTensor.get(),
                              m_CellBiasTensor.get(), m_OutputGateBiasTensor.get(), &output_state_in,
                              &cell_state_in, m_ScratchBuffer.get(), &output_state_out,
                              &cell_state_out, &output, lstm_param, activationLayerInfo,
                              cell_threshold, projection_threshold);
    }

    armcomputetensorutils::InitialiseArmComputeTensorEmpty(*m_ScratchBuffer);

    InitializeArmComputeClTensorData(*m_InputToForgetWeightsTensor,     m_Data.m_InputToForgetWeights);
    InitializeArmComputeClTensorData(*m_InputToCellWeightsTensor,       m_Data.m_InputToCellWeights);
    InitializeArmComputeClTensorData(*m_InputToOutputWeightsTensor,     m_Data.m_InputToOutputWeights);
    InitializeArmComputeClTensorData(*m_RecurrentToForgetWeightsTensor, m_Data.m_RecurrentToForgetWeights);
    InitializeArmComputeClTensorData(*m_RecurrentToCellWeightsTensor,   m_Data.m_RecurrentToCellWeights);
    InitializeArmComputeClTensorData(*m_RecurrentToOutputWeightsTensor, m_Data.m_RecurrentToOutputWeights);
    InitializeArmComputeClTensorData(*m_ForgetGateBiasTensor,           m_Data.m_ForgetGateBias);
    InitializeArmComputeClTensorData(*m_CellBiasTensor,                 m_Data.m_CellBias);
    InitializeArmComputeClTensorData(*m_OutputGateBiasTensor,           m_Data.m_OutputGateBias);

    if (!m_Data.m_Parameters.m_CifgEnabled)
    {
        InitializeArmComputeClTensorData(*m_InputToInputWeightsTensor, m_Data.m_InputToInputWeights);
        InitializeArmComputeClTensorData(*m_RecurrentToInputWeightsTensor, m_Data.m_RecurrentToInputWeights);
        if (m_Data.m_CellToInputWeights != nullptr)
        {
            InitializeArmComputeClTensorData(*m_CellToInputWeightsTensor, m_Data.m_CellToInputWeights);
        }
        InitializeArmComputeClTensorData(*m_InputGateBiasTensor, m_Data.m_InputGateBias);
    }

    if (m_Data.m_Parameters.m_ProjectionEnabled)
    {
        InitializeArmComputeClTensorData(*m_ProjectionWeightsTensor, m_Data.m_ProjectionWeights);
        if (m_Data.m_ProjectionBias != nullptr)
        {
            InitializeArmComputeClTensorData(*m_ProjectionBiasTensor, m_Data.m_ProjectionBias);
        }
    }

    if (m_Data.m_Parameters.m_PeepholeEnabled)
    {
        InitializeArmComputeClTensorData(*m_CellToForgetWeightsTensor, m_Data.m_CellToForgetWeights);
        InitializeArmComputeClTensorData(*m_CellToOutputWeightsTensor, m_Data.m_CellToOutputWeights);
    }

    if (m_Data.m_Parameters.m_LayerNormEnabled)
    {
        if (!m_Data.m_Parameters.m_CifgEnabled)
        {
            InitializeArmComputeClTensorData(*m_InputLayerNormWeightsTensor,  m_Data.m_InputLayerNormWeights);
        }

        InitializeArmComputeClTensorData(*m_ForgetLayerNormWeightsTensor, m_Data.m_ForgetLayerNormWeights);
        InitializeArmComputeClTensorData(*m_CellLayerNormWeightsTensor,   m_Data.m_CellLayerNormWeights);
        InitializeArmComputeClTensorData(*m_OutputLayerNormWeightsTensor, m_Data.m_OutputLayerNormWeights);
    }

    // Force Compute Library to perform the necessary copying and reshaping, after which
    // delete all the input tensors that will no longer be needed
    m_LstmLayer.prepare();
    FreeUnusedTensors();
}

void ClLstmFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClLstmFloatWorkload_Execute", GetGuid());
    RunClFunction(m_LstmLayer, CHECK_LOCATION());
}

arm_compute::Status ClLstmFloatWorkloadValidate(const TensorInfo& input, const TensorInfo& outputStateIn,
                                                const TensorInfo& cellStateIn, const TensorInfo& scratchBuffer,
                                                const TensorInfo& outputStateOut, const TensorInfo& cellStateOut,
                                                const TensorInfo& output, const LstmDescriptor& descriptor,
                                                const LstmInputParamsInfo& paramsInfo)
{
    arm_compute::LSTMParams<arm_compute::ITensorInfo> lstm_params_info;

    // The inputs and the outputs
    const arm_compute::TensorInfo aclInputInfo  = BuildArmComputeTensorInfo(input);
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
    const arm_compute::TensorInfo aclForgetGateBiasInfo = BuildArmComputeTensorInfo(paramsInfo.GetForgetGateBias());
    const arm_compute::TensorInfo aclCellBiasInfo = BuildArmComputeTensorInfo(paramsInfo.GetCellBias());
    const arm_compute::TensorInfo aclOutputGateBiasInfo = BuildArmComputeTensorInfo(paramsInfo.GetOutputGateBias());

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
        aclInputToInputWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetInputToInputWeights());
        aclRecurrentToInputWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetRecurrentToInputWeights());

        if (paramsInfo.m_CellToInputWeights != nullptr)
        {
            aclCellToInputWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetCellToInputWeights());
        }
        aclInputGateBiasInfo = BuildArmComputeTensorInfo(paramsInfo.GetInputGateBias());
        lstm_params_info.set_cifg_params(&aclInputToInputWeightsInfo, &aclRecurrentToInputWeightsInfo,
                                         paramsInfo.m_CellToInputWeights != nullptr ?
                                         &aclCellToInputWeightsInfo: nullptr,
                                         &aclInputGateBiasInfo);
    }

    if (descriptor.m_ProjectionEnabled)
    {
        aclProjectionWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetProjectionWeights());

        if (paramsInfo.m_ProjectionBias != nullptr)
        {
            aclProjectionBiasInfo = BuildArmComputeTensorInfo(paramsInfo.GetProjectionBias());
        }
        lstm_params_info.set_projection_params(&aclProjectionWeightsInfo,
                                               paramsInfo.m_ProjectionBias != nullptr ?
                                               &aclProjectionBiasInfo: nullptr);
    }

    if (descriptor.m_PeepholeEnabled)
    {
        aclCellToForgetWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetCellToForgetWeights());
        aclCellToOutputWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetCellToOutputWeights());
        lstm_params_info.set_peephole_params(&aclCellToForgetWeightsInfo, &aclCellToOutputWeightsInfo);
    }

    float cell_threshold = descriptor.m_ClippingThresCell;
    float projection_threshold = descriptor.m_ClippingThresProj;

    // for preparing the object for the class ActivationLayerInfo, we need to consider 5 situations
    arm_compute::ActivationLayerInfo activationLayerInfo =
        ConvertLstmActivationFuncToAclLayerInfo(descriptor.m_ActivationFunc);

    if (descriptor.m_LayerNormEnabled)
    {
        if (!descriptor.m_CifgEnabled)
        {
            aclInputLayerNormWeightsInfo  = BuildArmComputeTensorInfo(paramsInfo.GetInputLayerNormWeights());
        }

        aclForgetLayerNormWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetForgetLayerNormWeights());

        aclCellLayerNormWeightsInfo   = BuildArmComputeTensorInfo(paramsInfo.GetCellLayerNormWeights());

        aclOutputLayerNormWeightsInfo = BuildArmComputeTensorInfo(paramsInfo.GetOutputLayerNormWeights());

        lstm_params_info.set_layer_normalization_params(descriptor.m_CifgEnabled ?
                                                        nullptr : &aclInputLayerNormWeightsInfo,
                                                        &aclForgetLayerNormWeightsInfo,
                                                        &aclCellLayerNormWeightsInfo,
                                                        &aclOutputLayerNormWeightsInfo);
    }

    return arm_compute::CLLSTMLayer::validate(&aclInputInfo, &aclInputToForgetWeightsInfo,
                                              &aclInputToCellWeightsInfo,
                                              &aclInputToOutputWeightsInfo,
                                              &aclRecurrentToForgetWeightsInfo,
                                              &aclRecurrentToCellWeightsInfo,
                                              &aclRecurrentToOutputWeightsInfo,
                                              &aclForgetGateBiasInfo,
                                              &aclCellBiasInfo,
                                              &aclOutputGateBiasInfo,
                                              &aclOutputStateInInfo, &aclCellStateInInfo,
                                              &aclScratchBufferInfo, &aclOutputStateOutInfo,
                                              &aclCellStateOutInfo, &aclOutputInfo,
                                              lstm_params_info, activationLayerInfo,
                                              cell_threshold, projection_threshold);
}

void ClLstmFloatWorkload::FreeUnusedTensors()
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

void ClLstmFloatWorkload::ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot)
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
void ClLstmFloatWorkload::ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot)
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

void ClLstmFloatWorkload::Reconfigure()
{
    throw armnn::UnimplementedException("Reconfigure not implemented for this workload");
}

} //namespace armnn
