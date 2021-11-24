//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ResolveType.hpp>


#include <reference/workloads/Encoders.hpp>

#include <vector>

// Utility tenmplate to convert a collection of values to the correct type
template <armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
std::vector<T> ConvertToDataType(const std::vector<float>& input,
                                 const armnn::TensorInfo& inputTensorInfo)
{
    std::vector<T> output(input.size());
    auto outputTensorInfo = inputTensorInfo;
    outputTensorInfo.SetDataType(ArmnnType);

    std::unique_ptr<armnn::Encoder<float>> pOutputEncoder = armnn::MakeEncoder<float>(outputTensorInfo, output.data());
    armnn::Encoder<float>& rOutputEncoder = *pOutputEncoder;

    for (auto it = input.begin(); it != input.end(); ++it)
    {
        rOutputEncoder.Set(*it);
        ++rOutputEncoder;
    }
    return output;
}

// Utility tenmplate to convert a single value to the correct type
template <typename T>
T ConvertToDataType(const float& value,
                    const armnn::TensorInfo& tensorInfo)
{
    std::vector<T> output(1);
    std::unique_ptr<armnn::Encoder<float>> pEncoder = armnn::MakeEncoder<float>(tensorInfo, output.data());
    armnn::Encoder<float>& rEncoder = *pEncoder;
    rEncoder.Set(value);
    return output[0];
}
