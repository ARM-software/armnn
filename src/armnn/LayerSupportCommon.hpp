//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <armnn/DescriptorsFwd.hpp>
#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>

namespace armnn
{

template<typename Float32Func, typename Uint8Func, typename ... Params>
bool IsSupportedForDataTypeGeneric(std::string* reasonIfUnsupported,
                                   DataType dataType,
                                   Float32Func floatFuncPtr,
                                   Uint8Func uint8FuncPtr,
                                   Params&&... params)
{
    switch(dataType)
    {
        case DataType::Float32:
            return floatFuncPtr(reasonIfUnsupported, std::forward<Params>(params)...);
        case DataType::QuantisedAsymm8:
            return uint8FuncPtr(reasonIfUnsupported, std::forward<Params>(params)...);
        default:
            return false;
    }
}

template<typename ... Params>
bool TrueFunc(std::string* reasonIfUnsupported, Params&&... params)
{
    return true;
}

template<typename ... Params>
bool FalseFunc(std::string* reasonIfUnsupported, Params&&... params)
{
    return false;
}

template<typename ... Params>
bool FalseFuncF32(std::string* reasonIfUnsupported, Params&&... params)
{
    if (reasonIfUnsupported)
    {
        *reasonIfUnsupported = "Layer is not supported with float32 data type";
    }
    return false;
}

template<typename ... Params>
bool FalseFuncU8(std::string* reasonIfUnsupported, Params&&... params)
{
    if (reasonIfUnsupported)
    {
        *reasonIfUnsupported = "Layer is not supported with 8-bit data type";
    }
    return false;
}

}
