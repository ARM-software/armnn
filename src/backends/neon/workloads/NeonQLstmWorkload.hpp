//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Descriptors.hpp>
#include <armnn/LstmParams.hpp>
#include "NeonBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

#include "arm_compute/graph/Tensor.h"
#include "arm_compute/runtime/NEON/functions/NEQLSTMLayer.h"

namespace armnn
{

class NeonQLstmWorkload : public NeonBaseWorkload<QLstmQueueDescriptor>
{
public:
    NeonQLstmWorkload(const QLstmQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEQLSTMLayer m_QLstmLayer;

    std::unique_ptr<arm_compute::Tensor> m_InputToInputWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_InputToForgetWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_InputToCellWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_InputToOutputWeightsTensor;

    std::unique_ptr<arm_compute::Tensor> m_RecurrentToInputWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_RecurrentToForgetWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_RecurrentToCellWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_RecurrentToOutputWeightsTensor;

    std::unique_ptr<arm_compute::Tensor> m_CellToInputWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_CellToForgetWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_CellToOutputWeightsTensor;

    std::unique_ptr<arm_compute::Tensor> m_InputGateBiasTensor;
    std::unique_ptr<arm_compute::Tensor> m_ForgetGateBiasTensor;
    std::unique_ptr<arm_compute::Tensor> m_CellBiasTensor;
    std::unique_ptr<arm_compute::Tensor> m_OutputGateBiasTensor;

    std::unique_ptr<arm_compute::Tensor> m_ProjectionWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_ProjectionBiasTensor;

    std::unique_ptr<arm_compute::Tensor> m_InputLayerNormWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_ForgetLayerNormWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_CellLayerNormWeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_OutputLayerNormWeightsTensor;

    void FreeUnusedTensors();
};

arm_compute::Status NeonQLstmWorkloadValidate(const TensorInfo& input,
                                              const TensorInfo& cellStateIn,
                                              const TensorInfo& outputStateIn,
                                              const TensorInfo& cellStateOut,
                                              const TensorInfo& outputStateOut,
                                              const TensorInfo& output,
                                              const QLstmDescriptor& descriptor,
                                              const LstmInputParamsInfo& paramsInfo);

} //namespace armnn
