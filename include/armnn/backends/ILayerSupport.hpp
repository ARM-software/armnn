//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Deprecated.hpp>
#include <armnn/Descriptors.hpp>
#include <armnn/LstmParams.hpp>
#include <armnn/Optional.hpp>
#include <armnn/QuantizedLstmParams.hpp>

#include <cctype>
#include <functional>
#include <memory>
#include <vector>

namespace armnn
{

class TensorInfo;

class ILayerSupport
{
protected:
    ILayerSupport() {}
    virtual ~ILayerSupport() {}

public:
    virtual bool IsLayerSupported(const LayerType& type,
                                  const std::vector<TensorInfo>& infos,
                                  const BaseDescriptor& descriptor,
                                  const Optional<LstmInputParamsInfo>& lstmParamsInfo = EmptyOptional(),
                                  const Optional<QuantizedLstmInputParamsInfo>& quantizedLstmParamsInfo =
                                      EmptyOptional(),
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

}; // class ILayerSupport

using ILayerSupportSharedPtr = std::shared_ptr<ILayerSupport>;

} // namespace armnn
