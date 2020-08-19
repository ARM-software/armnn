//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SampleDynamicLayerSupport.hpp"

#include <InternalTypes.hpp>
#include <LayerSupportCommon.hpp>
#include <armnn/Types.hpp>

namespace sdb // sample dynamic backend
{

bool SampleDynamicLayerSupport::IsInputSupported(const armnn::TensorInfo& input,
                                                 armnn::Optional<std::string&> reasonIfUnsupported) const
{
    return true;
}

bool SampleDynamicLayerSupport::IsOutputSupported(const armnn::TensorInfo& output,
                                                  armnn::Optional<std::string&> reasonIfUnsupported) const
{
    return true;
}

bool SampleDynamicLayerSupport::IsAdditionSupported(const armnn::TensorInfo& input0,
                                                    const armnn::TensorInfo& input1,
                                                    const armnn::TensorInfo& output,
                                                    armnn::Optional<std::string&> reasonIfUnsupported) const
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

} // namespace sdb
