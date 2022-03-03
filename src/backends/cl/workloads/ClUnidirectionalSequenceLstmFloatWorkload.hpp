//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Descriptors.hpp>
#include <armnn/LstmParams.hpp>
#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadData.hpp>

#include <arm_compute/graph/Tensor.h>
#include <arm_compute/runtime/CL/functions/CLLSTMLayer.h>
#include <arm_compute/runtime/CL/functions/CLPermute.h>
#include <arm_compute/runtime/CL/functions/CLSplit.h>
#include <arm_compute/runtime/CL/functions/CLConcatenateLayer.h>

namespace armnn
{

class ClUnidirectionalSequenceLstmFloatWorkload : public FloatWorkload<UnidirectionalSequenceLstmQueueDescriptor>
{
public:
    ClUnidirectionalSequenceLstmFloatWorkload(const UnidirectionalSequenceLstmQueueDescriptor& descriptor,
                                              const WorkloadInfo& info,
                                              const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

private:

    //
    // ACL layers required to fully form a Unidirectional Sequence LSTM layer.
    //

    // permutation for input (only used when input is batch major)
    mutable std::unique_ptr<arm_compute::CLPermute> m_Permute1;
    mutable std::unique_ptr<arm_compute::IFunction> m_Splitter;
    mutable std::vector<std::unique_ptr<arm_compute::CLLSTMLayer>> m_Layers;
    mutable std::unique_ptr<arm_compute::CLConcatenateLayer> m_Concat;
    // permutation for output (only used when input is batch major)
    mutable std::unique_ptr<arm_compute::CLPermute> m_Permute2;

    //
    // ACL LSTM arm_compute::CLTensors.
    //
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

    std::unique_ptr<arm_compute::CLTensor> m_InputLayerNormWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_ForgetLayerNormWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_CellLayerNormWeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_OutputLayerNormWeightsTensor;

    //
    // Additional ACL arm_compute::CLTensors and std::vector<arm_compute::CLTensor>.
    // Required to perform splitting, concatenation and permutations.
    //
    arm_compute::CLTensor m_PermuteFirstOut;
    std::vector<arm_compute::CLTensor> m_SplitterOutputsTensors;
    std::vector<arm_compute::CLTensor> m_ConcatInputsTensors;
    std::vector<arm_compute::ICLTensor*> m_SplitterOutputs;
    std::vector<const arm_compute::ICLTensor*> m_ConcatInputs;
    arm_compute::CLTensor concat_out;

    void FreeUnusedTensors();
};

arm_compute::Status
ClUnidirectionalSequenceLstmFloatWorkloadValidate(const TensorInfo& input,
                                                  const TensorInfo& outputStateIn,
                                                  const TensorInfo& cellStateIn,
                                                  const TensorInfo& output,
                                                  const Optional<TensorInfo>& hiddenStateOutput,
                                                  const Optional<TensorInfo>& cellStateOutput,
                                                  const UnidirectionalSequenceLstmDescriptor& descriptor,
                                                  const LstmInputParamsInfo& paramsInfo);

} //namespace armnn
