//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <Layer.hpp>

namespace armnn
{

class ScopedTensorHandle;

struct QuantizedLstmParameters
{
    /// A unique pointer to represent 2D weights tensor with dimensions [outputSize, inputSize] (QAsymm8).
    std::shared_ptr<ConstTensorHandle> m_InputToInputWeights;
    /// A unique pointer to represent 2D weights tensor with dimensions [outputSize, inputSize] (QAsymm8).
    std::shared_ptr<ConstTensorHandle> m_InputToForgetWeights;
    /// A unique pointer to represent 2D weights tensor with dimensions [outputSize, inputSize] (QAsymm8).
    std::shared_ptr<ConstTensorHandle> m_InputToCellWeights;
    /// A unique pointer to represent 2D weights tensor with dimensions [outputSize, inputSize] (QAsymm8).
    std::shared_ptr<ConstTensorHandle> m_InputToOutputWeights;

    /// A unique pointer to represent 2D weights tensor with dimensions [outputSize, outputSize] (QAsymm8).
    std::shared_ptr<ConstTensorHandle> m_RecurrentToInputWeights;
    /// A unique pointer to represent 2D weights tensor with dimensions [outputSize, outputSize] (QAsymm8).
    std::shared_ptr<ConstTensorHandle> m_RecurrentToForgetWeights;
    /// A unique pointer to represent 2D weights tensor with dimensions [outputSize, outputSize] (QAsymm8).
    std::shared_ptr<ConstTensorHandle> m_RecurrentToCellWeights;
    /// A unique pointer to represent 2D weights tensor with dimensions [outputSize, outputSize] (QAsymm8).
    std::shared_ptr<ConstTensorHandle> m_RecurrentToOutputWeights;

    /// A unique pointer to represent 1D bias tensor with dimensions [outputSize] (int32).
    std::shared_ptr<ConstTensorHandle> m_InputGateBias;
    /// A unique pointer to represent 1D bias tensor with dimensions [outputSize] (int32).
    std::shared_ptr<ConstTensorHandle> m_ForgetGateBias;
    /// A unique pointer to represent 1D bias tensor with dimensions [outputSize] (int32).
    std::shared_ptr<ConstTensorHandle> m_CellBias;
    /// A unique pointer to represent 1D bias tensor with dimensions [outputSize] (int32).
    std::shared_ptr<ConstTensorHandle> m_OutputGateBias;
};

/// This layer represents a QuantizedLstm operation.
class QuantizedLstmLayer : public Layer
{
public:

    QuantizedLstmParameters m_QuantizedLstmParameters;

    /// Makes a workload for the QuantizedLstm type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    QuantizedLstmLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref QuantizedLstmLayer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    /// By default returns inputShapes if the number of inputs are equal to number of outputs,
    /// otherwise infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    void ExecuteStrategy(IStrategy& strategy) const override;

protected:
    /// Constructor to create a QuantizedLstmLayer.
    /// @param [in] name Optional name for the layer.
    QuantizedLstmLayer(const char* name);

    /// Default destructor
    ~QuantizedLstmLayer() = default;

    /// Retrieve the handles to the constant values stored by the layer.
    /// @return A vector of the constant tensors stored by this layer.
    Layer::ImmutableConstantTensors GetConstantTensorsByRef() const override;
};

} // namespace armnn
