//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Types.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/backends/ILayerSupport.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

namespace armnn
{

/// Default implementation of the ILayerSupport interface, Backends should implement this as a switch statement
/// for each of their LayerTypes calling their specific backend implementation of IsXXXLayerSupported.
bool ILayerSupport::IsLayerSupported(const LayerType& type,
                                     const std::vector<TensorInfo>& infos,
                                     const BaseDescriptor& descriptor,
                                     const Optional<LstmInputParamsInfo>& lstmParamsInfo,
                                     const Optional<QuantizedLstmInputParamsInfo>& quantizedLstmParamsInfo,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(infos);
    IgnoreUnused(descriptor);
    IgnoreUnused(lstmParamsInfo);
    IgnoreUnused(quantizedLstmParamsInfo);
    IgnoreUnused(reasonIfUnsupported);
    switch (type)
    {
        case LayerType::Map:
            return true;
        case LayerType::Unmap:
            return true;
        default:
            return false;
    }
}

}
