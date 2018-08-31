//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"
#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"

namespace armnn
{

class ClLstmFloat32Workload : public FloatWorkload<LstmQueueDescriptor>
{
public:
    ClLstmFloat32Workload(const LstmQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::CLLSTMLayer m_LstmLayer;

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

    std::unique_ptr<arm_compute::CLTensor> m_ScratchBuffer;

    void FreeUnusedTensors();
};

arm_compute::Status ClLstmFloat32WorkloadValidate(const TensorInfo& input, const TensorInfo& outputStateIn,
                                                  const TensorInfo& cellStateIn, const TensorInfo& scratchBuffer,
                                                  const TensorInfo& outputStateOut, const TensorInfo& cellStateOut,
                                                  const TensorInfo& output, const LstmDescriptor &descriptor,
                                                  const TensorInfo& inputToForgetWeights,
                                                  const TensorInfo& inputToCellWeights,
                                                  const TensorInfo& inputToOutputWeights,
                                                  const TensorInfo& recurrentToForgetWeights,
                                                  const TensorInfo& recurrentToCellWeights,
                                                  const TensorInfo& recurrentToOutputWeights,
                                                  const TensorInfo& forgetGateBias, const TensorInfo& cellBias,
                                                  const TensorInfo& outputGateBias,
                                                  const TensorInfo* inputToInputWeights,
                                                  const TensorInfo* recurrentToInputWeights,
                                                  const TensorInfo* cellToInputWeights,
                                                  const TensorInfo* inputGateBias,
                                                  const TensorInfo* projectionWeights,
                                                  const TensorInfo* projectionBias,
                                                  const TensorInfo* cellToForgetWeights,
                                                  const TensorInfo* cellToOutputWeights);
} //namespace armnn
