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
    bool IsLayerSupported(const LayerType& type,
                          const std::vector<TensorInfo>& infos,
                          const BaseDescriptor& /*descriptor*/,
                          const Optional<LstmInputParamsInfo>& /*lstmParamsInfo*/,
                          const Optional<QuantizedLstmInputParamsInfo>& /*quantizedLstmParamsInfo*/,
                          Optional<std::string&> reasonIfUnsupported) const override
    {
        switch(type)
        {
            case LayerType::Addition:
                return IsAdditionSupported(infos[0], infos[1], infos[2], reasonIfUnsupported);
            case LayerType::Input:
                return IsInputSupported(infos[0], reasonIfUnsupported);
            case LayerType::Output:
                return IsOutputSupported(infos[0], reasonIfUnsupported);
            case LayerType::MemCopy:
                return LayerSupportBase::IsMemCopySupported(infos[0], infos[1], reasonIfUnsupported);
            case LayerType::MemImport:
                return LayerSupportBase::IsMemImportSupported(infos[0], infos[1], reasonIfUnsupported);
            default:
                return false;
        }
    }

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
