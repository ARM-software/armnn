//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

/// This layer represents a split operation.
class SplitterLayer : public LayerWithParameters<ViewsDescriptor>
{
public:
    /// Makes a workload for the Splitter type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Set the outputs to be appropriate sub tensors of the input if sub tensors are supported
    /// otherwise creates tensor handlers.
    /// @param [in] registry Contains all the registered tensor handle factories available for use.
    /// @param [in] factory The workload factory which will create the workload.
    /// @param [in] IsMemoryManaged Determine whether or not to assign a memory manager during creation
    virtual void CreateTensorHandles(const TensorHandleFactoryRegistry& registry,
                                     const IWorkloadFactory& factory,
                                     const bool IsMemoryManaged = true) override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    SplitterLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref SplitterLayer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    /// By default returns inputShapes if the number of inputs are equal to number of outputs,
    /// otherwise infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    void ExecuteStrategy(IStrategy& strategy) const override;


protected:
    /// Constructor to create a SplitterLayer.
    /// @param [in] param ViewsDescriptor to configure the splitter operation.
    /// @param [in] name Optional name for the layer.
    SplitterLayer(const ViewsDescriptor& param, const char* name);

    /// Default destructor
    ~SplitterLayer() = default;

private:
    template <typename FactoryType>
    void CreateTensors(const TensorHandleFactoryRegistry& registry, const FactoryType& factory, bool isMemoryManaged);
};

} // namespace
