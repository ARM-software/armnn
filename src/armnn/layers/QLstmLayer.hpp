//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class ScopedTensorHandle;

struct QLstmBasicParameters
{
    /// A unique pointer to represent 2D weights tensor with dimensions [num_units, inputSize] (QSymmS8).
    std::shared_ptr<ConstTensorHandle> m_InputToForgetWeights;
    /// A unique pointer to represent 2D weights tensor with dimensions [num_units, inputSize] (QSymmS8).
    std::shared_ptr<ConstTensorHandle> m_InputToCellWeights;
    /// A unique pointer to represent 2D weights tensor with dimensions [num_units, inputSize] (QSymmS8).
    std::shared_ptr<ConstTensorHandle> m_InputToOutputWeights;

    /// A unique pointer to represent 2D weights tensor with dimensions [num_units, outputSize] (QSymmS8).
    std::shared_ptr<ConstTensorHandle> m_RecurrentToForgetWeights;
    /// A unique pointer to represent 2D weights tensor with dimensions [num_units, outputSize] (QSymmS8).
    std::shared_ptr<ConstTensorHandle> m_RecurrentToCellWeights;
    /// A unique pointer to represent 2D weights tensor with dimensions [num_units, outputSize] (QSymmS8).
    std::shared_ptr<ConstTensorHandle> m_RecurrentToOutputWeights;

    /// A unique pointer to represent 1D bias tensor with dimensions [num_units] (int32).
    std::shared_ptr<ConstTensorHandle> m_ForgetGateBias;
    /// A unique pointer to represent 1D bias tensor with dimensions [num_units] (int32).
    std::shared_ptr<ConstTensorHandle> m_CellBias;
    /// A unique pointer to represent 1D bias tensor with dimensions [num_units] (int32).
    std::shared_ptr<ConstTensorHandle> m_OutputGateBias;
};

struct QLstmOptProjectionParameters
{
    /// A unique pointer to represent 2D weights tensor with dimensions [output_size, num_units] (QSymmS8).
    std::shared_ptr<ConstTensorHandle> m_ProjectionWeights;
    /// A unique pointer to represent 1D weights tensor with dimensions [output_size] (int32).
    std::shared_ptr<ConstTensorHandle> m_ProjectionBias;
};

struct QLstmOptPeepholeParameters
{
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units] (QSymmS16).
    std::shared_ptr<ConstTensorHandle> m_CellToInputWeights;
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units] (QSymmS16).
    std::shared_ptr<ConstTensorHandle> m_CellToForgetWeights;
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units] (QSymmS16).
    std::shared_ptr<ConstTensorHandle> m_CellToOutputWeights;
};

struct QLstmOptCifgParameters
{
    /// A unique pointer to represent 2D weights tensor with dimensions [input_size, num_units] (QSymmS8).
    std::shared_ptr<ConstTensorHandle> m_InputToInputWeights;
    /// A unique pointer to represent 2D weights tensor with dimensions [input_size, num_units] (QSymmS8).
    std::shared_ptr<ConstTensorHandle> m_RecurrentToInputWeights;
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units] (int32).
    std::shared_ptr<ConstTensorHandle> m_InputGateBias;
};

struct QLstmOptLayerNormParameters
{
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units] (QSymmS16).
    std::shared_ptr<ConstTensorHandle> m_InputLayerNormWeights;
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units] (QSymmS16).
    std::shared_ptr<ConstTensorHandle> m_ForgetLayerNormWeights;
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units] (QSymmS16).
    std::shared_ptr<ConstTensorHandle> m_CellLayerNormWeights;
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units] (QSymmS16).
    std::shared_ptr<ConstTensorHandle> m_OutputLayerNormWeights;
};

/// This layer represents a QLstm operation.
class QLstmLayer : public LayerWithParameters<QLstmDescriptor>
{
public:

    QLstmBasicParameters m_BasicParameters;
    QLstmOptCifgParameters m_CifgParameters;
    QLstmOptProjectionParameters m_ProjectionParameters;
    QLstmOptPeepholeParameters m_PeepholeParameters;
    QLstmOptLayerNormParameters m_LayerNormParameters;

    /// Makes a workload for the QLstm type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    QLstmLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref QLstmLayer.
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
    /// Constructor to create a QLstmLayer.
    /// @param [in] name Optional name for the layer.
    QLstmLayer(const QLstmDescriptor& param, const char* name);

    /// Default destructor
    ~QLstmLayer() = default;

    /// Retrieve the handles to the constant values stored by the layer.
    /// @return A vector of the constant tensors stored by this layer.
    Layer::ConstantTensors GetConstantTensorsByRef() override;
};

} // namespace armnn
