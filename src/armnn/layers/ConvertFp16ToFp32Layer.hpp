//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <Layer.hpp>

namespace armnn
{

/// This layer converts data type Float 16 to Float 32.
class ConvertFp16ToFp32Layer : public Layer
{
public:
    /// Makes a workload for the ConvertFp16ToFp32 type.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    ConvertFp16ToFp32Layer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref ConvertFp16ToFp32Layer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    void ExecuteStrategy(IStrategy& strategy) const override;

protected:
    /// Constructor to create a ConvertFp16ToFp32Layer.
    /// @param [in] name Optional name for the layer.
    ConvertFp16ToFp32Layer(const char* name);

    /// Default destructor
    ~ConvertFp16ToFp32Layer() = default;
};

} // namespace
