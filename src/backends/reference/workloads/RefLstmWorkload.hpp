//
// Copyright © 2022, 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/TypesUtils.hpp>

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

class RefLstmWorkload : public RefBaseWorkload<LstmQueueDescriptor>
{
public:
    explicit RefLstmWorkload(const LstmQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
    std::unique_ptr<ScopedTensorHandle> m_InputToInputWeightsTensor;
    std::unique_ptr<ScopedTensorHandle> m_InputToForgetWeightsTensor;
    std::unique_ptr<ScopedTensorHandle> m_InputToCellWeightsTensor;
    std::unique_ptr<ScopedTensorHandle> m_InputToOutputWeightsTensor;
    std::unique_ptr<ScopedTensorHandle> m_RecurrentToInputWeightsTensor;
    std::unique_ptr<ScopedTensorHandle> m_RecurrentToForgetWeightsTensor;
    std::unique_ptr<ScopedTensorHandle> m_RecurrentToCellWeightsTensor;
    std::unique_ptr<ScopedTensorHandle> m_RecurrentToOutputWeightsTensor;
    std::unique_ptr<ScopedTensorHandle> m_CellToInputWeightsTensor;
    std::unique_ptr<ScopedTensorHandle> m_CellToForgetWeightsTensor;
    std::unique_ptr<ScopedTensorHandle> m_CellToOutputWeightsTensor;
    std::unique_ptr<ScopedTensorHandle> m_InputGateBiasTensor;
    std::unique_ptr<ScopedTensorHandle> m_ForgetGateBiasTensor;
    std::unique_ptr<ScopedTensorHandle> m_CellBiasTensor;
    std::unique_ptr<ScopedTensorHandle> m_OutputGateBiasTensor;
    std::unique_ptr<ScopedTensorHandle> m_ProjectionWeightsTensor;
    std::unique_ptr<ScopedTensorHandle> m_ProjectionBiasTensor;
    std::unique_ptr<ScopedTensorHandle> m_InputLayerNormWeights;
    std::unique_ptr<ScopedTensorHandle> m_ForgetLayerNormWeights;
    std::unique_ptr<ScopedTensorHandle> m_CellLayerNormWeights;
    std::unique_ptr<ScopedTensorHandle> m_OutputLayerNormWeights;

    float m_LayerNormEpsilon = static_cast<float>(1e-8);
};

} //namespace armnn
