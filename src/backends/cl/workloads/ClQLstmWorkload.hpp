//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Descriptors.hpp>
#include <armnn/LstmParams.hpp>
#include "ClBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

#include "arm_compute/graph/Tensor.h"
#include "arm_compute/runtime/CL/functions/CLQLSTMLayer.h"

namespace armnn
{

class ClQLstmWorkload : public ClBaseWorkload<QLstmQueueDescriptor>
{
public:
    ClQLstmWorkload(const QLstmQueueDescriptor& descriptor,
                    const WorkloadInfo& info,
                    const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLQLSTMLayer m_QLstmLayer;

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

    void FreeUnusedTensors();
};

arm_compute::Status ClQLstmWorkloadValidate(const TensorInfo& input,
                                            const TensorInfo& cellStateIn,
                                            const TensorInfo& outputStateIn,
                                            const TensorInfo& cellStateOut,
                                            const TensorInfo& outputStateOut,
                                            const TensorInfo& output,
                                            const QLstmDescriptor& descriptor,
                                            const LstmInputParamsInfo& paramsInfo);
} //namespace armnn
