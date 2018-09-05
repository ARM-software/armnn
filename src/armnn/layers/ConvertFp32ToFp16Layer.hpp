//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <Layer.hpp>

namespace armnn
{

class ConvertFp32ToFp16Layer : public Layer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
                                                      const IWorkloadFactory& factory) const override;

    ConvertFp32ToFp16Layer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    ConvertFp32ToFp16Layer(const char* name);
    ~ConvertFp32ToFp16Layer() = default;
};

} // namespace
