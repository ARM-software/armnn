//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"
#include "LstmParameters.hpp"

namespace armnn
{

class ScopedTensorHandle;

/// This layer represents a LSTM operation.
class LstmLayer : public LayerWithParameters<LstmDescriptor>
{
public:

    LstmBasicParameters m_BasicParameters;
    LstmOptCifgParameters m_CifgParameters;
    LstmOptProjectionParameters m_ProjectionParameters;
    LstmOptPeepholeParameters m_PeepholeParameters;
    LstmOptLayerNormParameters m_LayerNormParameters;

    /// Makes a workload for the LSTM type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    LstmLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref LstmLayer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    /// By default returns inputShapes if the number of inputs are equal to number of outputs,
    /// otherwise infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    void Accept(ILayerVisitor& visitor) const override;
    ARMNN_NO_DEPRECATE_WARN_END

    void ExecuteStrategy(IStrategy& strategy) const override;

protected:
    /// Constructor to create a LstmLayer.
    /// @param [in] param LstmDescriptor to configure the lstm operation.
    /// @param [in] name Optional name for the layer.
    LstmLayer(const LstmDescriptor& param, const char* name);

    /// Default destructor
    ~LstmLayer() = default;

    /// Retrieve the handles to the constant values stored by the layer.
    /// @return A vector of the constant tensors stored by this layer.
    Layer::ConstantTensors GetConstantTensorsByRef() override;
};

} // namespace
