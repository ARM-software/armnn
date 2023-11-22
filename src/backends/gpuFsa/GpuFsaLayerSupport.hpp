//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <backendsCommon/LayerSupportBase.hpp>
#include <backendsCommon/LayerSupportRules.hpp>

namespace armnn
{

class GpuFsaLayerSupport : public ILayerSupport
{
public:
    bool IsLayerSupported(const LayerType& type,
                          const std::vector<TensorInfo>& infos,
                          const BaseDescriptor& descriptor,
                          const Optional<LstmInputParamsInfo>& lstmParamsInfo,
                          const Optional<QuantizedLstmInputParamsInfo>&,
                          Optional<std::string&> reasonIfUnsupported) const override;
};

} // namespace armnn