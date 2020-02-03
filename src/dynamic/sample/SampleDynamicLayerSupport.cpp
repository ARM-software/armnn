//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SampleDynamicLayerSupport.hpp"

#include <InternalTypes.hpp>
#include <LayerSupportCommon.hpp>
#include <armnn/Types.hpp>

namespace armnn
{

bool SampleDynamicLayerSupport::IsInputSupported(const TensorInfo& input,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    return true;
}

bool SampleDynamicLayerSupport::IsOutputSupported(const TensorInfo& output,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    return true;
}

bool SampleDynamicLayerSupport::IsAdditionSupported(const TensorInfo& input0,
                                                    const TensorInfo& input1,
                                                    const TensorInfo& output,
                                                    Optional<std::string&> reasonIfUnsupported) const
{

    if (input0.GetDataType() != armnn::DataType::Float32)
    {
        return false;
    }

    if (input0.GetDataType() != input1.GetDataType())
    {
        return false;
    }

    if (input0.GetDataType() != output.GetDataType())
    {
        return false;
    }

    return true;
}

} // namespace armnn
