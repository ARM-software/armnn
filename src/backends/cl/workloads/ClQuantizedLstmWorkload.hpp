//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <armnn/QuantizedLstmParams.hpp>
#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadData.hpp>

#include <arm_compute/runtime/CL/functions/CLLSTMLayerQuantized.h>

namespace armnn
{

arm_compute::Status ClQuantizedLstmWorkloadValidate(const TensorInfo& input, const TensorInfo& previousCellStateIn,
                                                    const TensorInfo& previousOutputIn, const TensorInfo& cellStateOut,
                                                    const TensorInfo& output,
                                                    const QuantizedLstmInputParamsInfo& paramsInfo);

class ClQuantizedLstmWorkload : public ClBaseWorkload<QuantizedLstmQueueDescriptor>
{
public:
    ClQuantizedLstmWorkload(const QuantizedLstmQueueDescriptor& descriptor,
                            const WorkloadInfo& info,
                            const arm_compute::CLCompileContext& clCompileContext);
    void Execute() const override;

private:
    mutable arm_compute::CLLSTMLayerQuantized m_QuantizedLstmLayer;

    std::unique_ptr<arm_compute::CLTensor> m_InputToInputWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_InputToForgetWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_InputToCellWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_InputToOutputWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_RecurrentToInputWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_RecurrentToForgetWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_RecurrentToCellWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_RecurrentToOutputWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_InputGateBiasTensor;
    std::unique_ptr<arm_compute::CLTensor> m_ForgetGateBiasTensor;
    std::unique_ptr<arm_compute::CLTensor> m_CellBiasTensor;
    std::unique_ptr<arm_compute::CLTensor> m_OutputGateBiasTensor;

    void FreeUnusedTensors();
};

} //namespace armnn


