//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Descriptors.hpp>
#include <armnn/LstmParams.hpp>
#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadData.hpp>

#include <arm_compute/runtime/CL/functions/CLLSTMLayer.h>

namespace armnn
{

class ClLstmFloatWorkload : public FloatWorkload<LstmQueueDescriptor>
{
public:
    ClLstmFloatWorkload(const LstmQueueDescriptor& descriptor,
                        const WorkloadInfo& info,
                        const arm_compute::CLCompileContext& clCompileContext);
    void Execute() const override;
    // Replace input tensor handle with the given TensorHandle
    void ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override;

    // Replace output tensor handle with the given TensorHandle
    void ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override;
private:
    mutable arm_compute::CLLSTMLayer m_LstmLayer;
    virtual void Reconfigure();

    std::unique_ptr<arm_compute::CLTensor> m_InputToInputWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_InputToForgetWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_InputToCellWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_InputToOutputWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_RecurrentToInputWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_RecurrentToForgetWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_RecurrentToCellWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_RecurrentToOutputWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_CellToInputWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_CellToForgetWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_CellToOutputWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_InputGateBiasTensor;
    std::unique_ptr<arm_compute::CLTensor> m_ForgetGateBiasTensor;
    std::unique_ptr<arm_compute::CLTensor> m_CellBiasTensor;
    std::unique_ptr<arm_compute::CLTensor> m_OutputGateBiasTensor;
    std::unique_ptr<arm_compute::CLTensor> m_ProjectionWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_ProjectionBiasTensor;
    std::unique_ptr<arm_compute::CLTensor> m_InputLayerNormWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_ForgetLayerNormWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_CellLayerNormWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_OutputLayerNormWeightsTensor;

    std::unique_ptr<arm_compute::CLTensor> m_ScratchBuffer;

    void FreeUnusedTensors();
};

arm_compute::Status ClLstmFloatWorkloadValidate(const TensorInfo& input, const TensorInfo& outputStateIn,
                                                const TensorInfo& cellStateIn, const TensorInfo& scratchBuffer,
                                                const TensorInfo& outputStateOut, const TensorInfo& cellStateOut,
                                                const TensorInfo& output, const LstmDescriptor &descriptor,
                                                const LstmInputParamsInfo& paramsInfo);
} //namespace armnn
