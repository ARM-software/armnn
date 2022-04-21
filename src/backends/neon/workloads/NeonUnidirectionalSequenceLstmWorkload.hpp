//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Descriptors.hpp>
#include <armnn/LstmParams.hpp>
#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include "NeonBaseWorkload.hpp"

#include "arm_compute/runtime/NEON/functions/NEQLSTMLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPermute.h"
#include "arm_compute/runtime/NEON/functions/NESplit.h"
#include "arm_compute/runtime/NEON/functions/NEConcatenateLayer.h"

namespace armnn
{

class NeonUnidirectionalSequenceLstmWorkload : public NeonBaseWorkload<UnidirectionalSequenceLstmQueueDescriptor>
{
public:
    NeonUnidirectionalSequenceLstmWorkload(const UnidirectionalSequenceLstmQueueDescriptor& descriptor,
                                           const WorkloadInfo& info);
    virtual void Execute() const override;

private:

    //
    // ACL layers required to fully form a Unidirectional Sequence LSTM layer.
    //
    mutable std::unique_ptr<arm_compute::NEPermute> m_Permute1;
    mutable std::unique_ptr<arm_compute::IFunction> m_Splitter;
    mutable std::vector<std::unique_ptr<arm_compute::NEQLSTMLayer>> m_Layers;
    mutable std::unique_ptr<arm_compute::NEConcatenateLayer> m_Concat;
    mutable std::unique_ptr<arm_compute::NEPermute> m_Permute2;

    //
    // ACL LSTM arm_compute::Tensors.
    //
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

    //
    // Additional ACL arm_compute::Tensors and std::vector<arm_compute::Tensor>.
    // Required to perform splitting, concatenation and permutations.
    //
    arm_compute::Tensor m_PermuteFirstOut;
    std::vector<arm_compute::Tensor> m_SplitterOutputsTensors;
    std::vector<arm_compute::Tensor> m_ConcatInputsTensors;
    std::vector<arm_compute::ITensor*> m_SplitterOutputs;
    std::vector<const arm_compute::ITensor*> m_ConcatInputs;
    arm_compute::Tensor concat_out;

    void FreeUnusedTensors();
};

arm_compute::Status
NeonUnidirectionalSequenceLstmWorkloadValidate(const TensorInfo& input,
                                               const TensorInfo& outputStateIn,
                                               const TensorInfo& cellStateIn,
                                               const TensorInfo& outputStateOut,
                                               const TensorInfo& cellStateOut,
                                               const TensorInfo& output,
                                               const UnidirectionalSequenceLstmDescriptor& descriptor,
                                               const LstmInputParamsInfo& paramsInfo);

} //namespace armnn
