//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/utility/IgnoreUnused.hpp>

#include <backendsCommon/LayerSupportBase.hpp>

namespace armnn
{

class MockImportLayerSupport : public LayerSupportBase
{
public:
    bool IsAdditionSupported(const TensorInfo& input0,
                             const TensorInfo& input1,
                             const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override
    {
        IgnoreUnused(input0);
        IgnoreUnused(input1);
        IgnoreUnused(output);
        IgnoreUnused(reasonIfUnsupported);
        return true;
    }

    bool IsInputSupported(const TensorInfo& input,
                          Optional<std::string&> reasonIfUnsupported) const override
    {
        IgnoreUnused(input);
        IgnoreUnused(reasonIfUnsupported);
        return true;
    }

    bool IsOutputSupported(const TensorInfo& output,
                           Optional<std::string&> reasonIfUnsupported) const override
    {
        IgnoreUnused(output);
        IgnoreUnused(reasonIfUnsupported);
        return true;
    }
};

} // namespace armnn
