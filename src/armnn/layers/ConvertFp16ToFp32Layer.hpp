//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <Layer.hpp>

namespace armnn
{

class ConvertFp16ToFp32Layer : public Layer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
                                                      const IWorkloadFactory& factory) const override;

    ConvertFp16ToFp32Layer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    ConvertFp16ToFp32Layer(const char* name);
    ~ConvertFp16ToFp32Layer() = default;
};

} // namespace
