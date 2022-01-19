//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/QuantizedLstmParams.hpp>
#include "NeonBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

#include <arm_compute/graph/Tensor.h>
#include <arm_compute/runtime/NEON/functions/NELSTMLayerQuantized.h>

namespace armnn
{

class NeonQuantizedLstmWorkload : public NeonBaseWorkload<QuantizedLstmQueueDescriptor>
{
public:
    using BaseWorkload<QuantizedLstmQueueDescriptor>::m_Data;
    NeonQuantizedLstmWorkload(const QuantizedLstmQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NELSTMLayerQuantized m_QuantizedLstmLayer;

    std::unique_ptr<arm_compute::Tensor> m_InputToInputWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_InputToForgetWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_InputToCellWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_InputToOutputWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_RecurrentToInputWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_RecurrentToForgetWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_RecurrentToCellWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_RecurrentToOutputWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_InputGateBiasTensor;
    std::unique_ptr<arm_compute::Tensor> m_ForgetGateBiasTensor;
    std::unique_ptr<arm_compute::Tensor> m_CellBiasTensor;
    std::unique_ptr<arm_compute::Tensor> m_OutputGateBiasTensor;
    std::unique_ptr<arm_compute::Tensor> m_CellStateInTensor;
    std::unique_ptr<arm_compute::Tensor> m_OutputStateInTensor;
    std::unique_ptr<arm_compute::Tensor> m_CellStateOutTensor;

    void FreeUnusedTensors();
};

arm_compute::Status NeonQuantizedLstmWorkloadValidate(const TensorInfo& input,
                                                      const TensorInfo& outputStateIn,
                                                      const TensorInfo& cellStateIn,
                                                      const TensorInfo& outputStateOut,
                                                      const TensorInfo& cellStateOut,
                                                      const QuantizedLstmInputParamsInfo& paramsInfo);

} //namespace armnn
