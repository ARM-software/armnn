//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/TypesUtils.hpp>

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefLstmWorkload : public BaseWorkload<LstmQueueDescriptor>
{
public:
    explicit RefLstmWorkload(const LstmQueueDescriptor& descriptor, const WorkloadInfo& info);

    virtual void Execute() const override;

private:
    std::unique_ptr<ScopedCpuTensorHandle> m_InputToInputWeightsTensor;
    std::unique_ptr<ScopedCpuTensorHandle> m_InputToForgetWeightsTensor;
    std::unique_ptr<ScopedCpuTensorHandle> m_InputToCellWeightsTensor;
    std::unique_ptr<ScopedCpuTensorHandle> m_InputToOutputWeightsTensor;
    std::unique_ptr<ScopedCpuTensorHandle> m_RecurrentToInputWeightsTensor;
    std::unique_ptr<ScopedCpuTensorHandle> m_RecurrentToForgetWeightsTensor;
    std::unique_ptr<ScopedCpuTensorHandle> m_RecurrentToCellWeightsTensor;
    std::unique_ptr<ScopedCpuTensorHandle> m_RecurrentToOutputWeightsTensor;
    std::unique_ptr<ScopedCpuTensorHandle> m_CellToInputWeightsTensor;
    std::unique_ptr<ScopedCpuTensorHandle> m_CellToForgetWeightsTensor;
    std::unique_ptr<ScopedCpuTensorHandle> m_CellToOutputWeightsTensor;
    std::unique_ptr<ScopedCpuTensorHandle> m_InputGateBiasTensor;
    std::unique_ptr<ScopedCpuTensorHandle> m_ForgetGateBiasTensor;
    std::unique_ptr<ScopedCpuTensorHandle> m_CellBiasTensor;
    std::unique_ptr<ScopedCpuTensorHandle> m_OutputGateBiasTensor;
    std::unique_ptr<ScopedCpuTensorHandle> m_ProjectionWeightsTensor;
    std::unique_ptr<ScopedCpuTensorHandle> m_ProjectionBiasTensor;
    std::unique_ptr<ScopedCpuTensorHandle> m_InputLayerNormWeights;
    std::unique_ptr<ScopedCpuTensorHandle> m_ForgetLayerNormWeights;
    std::unique_ptr<ScopedCpuTensorHandle> m_CellLayerNormWeights;
    std::unique_ptr<ScopedCpuTensorHandle> m_OutputLayerNormWeights;

    float m_LayerNormEpsilon = static_cast<float>(1e-8);
};

} //namespace armnn
