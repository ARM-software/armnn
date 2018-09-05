//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class ScopedCpuTensorHandle;

struct LstmOptCifgParameters
{
    std::unique_ptr<ScopedCpuTensorHandle> m_InputToInputWeights;
    std::unique_ptr<ScopedCpuTensorHandle> m_RecurrentToInputWeights;
    std::unique_ptr<ScopedCpuTensorHandle> m_CellToInputWeights;
    std::unique_ptr<ScopedCpuTensorHandle> m_InputGateBias;
};

struct LstmOptProjectionParameters
{
    std::unique_ptr<ScopedCpuTensorHandle> m_ProjectionWeights;
    std::unique_ptr<ScopedCpuTensorHandle> m_ProjectionBias;
};

struct LstmOptPeepholeParameters
{
    std::unique_ptr<ScopedCpuTensorHandle> m_CellToForgetWeights;
    std::unique_ptr<ScopedCpuTensorHandle> m_CellToOutputWeights;
};

struct LstmBasicParameters
{
    std::unique_ptr<ScopedCpuTensorHandle> m_InputToForgetWeights;
    std::unique_ptr<ScopedCpuTensorHandle> m_InputToCellWeights;
    std::unique_ptr<ScopedCpuTensorHandle> m_InputToOutputWeights;
    std::unique_ptr<ScopedCpuTensorHandle> m_RecurrentToForgetWeights;
    std::unique_ptr<ScopedCpuTensorHandle> m_RecurrentToCellWeights;
    std::unique_ptr<ScopedCpuTensorHandle> m_RecurrentToOutputWeights;
    std::unique_ptr<ScopedCpuTensorHandle> m_ForgetGateBias;
    std::unique_ptr<ScopedCpuTensorHandle> m_CellBias;
    std::unique_ptr<ScopedCpuTensorHandle> m_OutputGateBias;
};

class LstmLayer : public LayerWithParameters<LstmDescriptor>
{
public:

    LstmBasicParameters m_BasicParameters;
    LstmOptCifgParameters m_CifgParameters;
    LstmOptProjectionParameters m_ProjectionParameters;
    LstmOptPeepholeParameters m_PeepholeParameters;

    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;
    LstmLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

protected:
    LstmLayer(const LstmDescriptor& param, const char* name);
    ~LstmLayer() = default;

    Layer::ConstantTensors GetConstantTensorsByRef() override;
};

} // namespace
