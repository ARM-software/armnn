//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/DescriptorsFwd.hpp>
#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Optional.hpp>

namespace armnn
{

template<typename T, typename V>
void SetValueChecked(Optional<T&> optionalRef, V&& val)
{
    if (optionalRef)
    {
        optionalRef.value() = val;
    }
}

template<typename Float16Func, typename Float32Func, typename Uint8Func, typename Int32Func, typename BooleanFunc,
         typename ... Params>
bool IsSupportedForDataTypeGeneric(Optional<std::string&> reasonIfUnsupported,
                                   DataType dataType,
                                   Float16Func float16FuncPtr,
                                   Float32Func float32FuncPtr,
                                   Uint8Func uint8FuncPtr,
                                   Int32Func int32FuncPtr,
                                   BooleanFunc booleanFuncPtr,
                                   Params&&... params)
{
    switch(dataType)
    {
        case DataType::Float16:
            return float16FuncPtr(reasonIfUnsupported, std::forward<Params>(params)...);
        case DataType::Float32:
            return float32FuncPtr(reasonIfUnsupported, std::forward<Params>(params)...);
        case DataType::QAsymmU8:
            return uint8FuncPtr(reasonIfUnsupported, std::forward<Params>(params)...);
        case DataType::Signed32:
            return int32FuncPtr(reasonIfUnsupported, std::forward<Params>(params)...);
        case DataType::Boolean:
            return booleanFuncPtr(reasonIfUnsupported, std::forward<Params>(params)...);
        default:
            return false;
    }
}

template<typename ... Params>
bool TrueFunc(Optional<std::string&> reasonIfUnsupported, Params&&... params)
{
    IgnoreUnused(reasonIfUnsupported);
    IgnoreUnused(params...);
    return true;
}

template<typename ... Params>
bool FalseFunc(Optional<std::string&> reasonIfUnsupported, Params&&... params)
{
    IgnoreUnused(reasonIfUnsupported);
    IgnoreUnused(params...);
    return false;
}

template<typename ... Params>
bool FalseFuncF16(Optional<std::string&> reasonIfUnsupported, Params&&... params)
{
    IgnoreUnused(params...);
    SetValueChecked(reasonIfUnsupported, "Layer is not supported with float16 data type");
    return false;
}

template<typename ... Params>
bool FalseFuncF32(Optional<std::string&> reasonIfUnsupported, Params&&... params)
{
    IgnoreUnused(params...);
    SetValueChecked(reasonIfUnsupported, "Layer is not supported with float32 data type");
    return false;
}

template<typename ... Params>
bool FalseFuncU8(Optional<std::string&> reasonIfUnsupported, Params&&... params)
{
    IgnoreUnused(params...);
    SetValueChecked(reasonIfUnsupported, "Layer is not supported with 8-bit data type");
    return false;
}

template<typename ... Params>
bool FalseFuncI32(Optional<std::string&> reasonIfUnsupported, Params&&... params)
{
    IgnoreUnused(params...);
    SetValueChecked(reasonIfUnsupported, "Layer is not supported with int32 data type");
    return false;
}

template<typename ... Params>
bool FalseInputFuncF32(Optional<std::string&> reasonIfUnsupported, Params&&... params)
{
    IgnoreUnused(params...);
    SetValueChecked(reasonIfUnsupported, "Layer is not supported with float32 data type input");
    return false;
}

template<typename ... Params>
bool FalseInputFuncF16(Optional<std::string&> reasonIfUnsupported, Params&&... params)
{
    IgnoreUnused(params...);
    SetValueChecked(reasonIfUnsupported, "Layer is not supported with float16 data type input");
    return false;
}

template<typename ... Params>
bool FalseOutputFuncF32(Optional<std::string&> reasonIfUnsupported, Params&&... params)
{
    IgnoreUnused(params...);
    SetValueChecked(reasonIfUnsupported, "Layer is not supported with float32 data type output");
    return false;
}

template<typename ... Params>
bool FalseOutputFuncF16(Optional<std::string&> reasonIfUnsupported, Params&&... params)
{
    IgnoreUnused(params...);
    SetValueChecked(reasonIfUnsupported, "Layer is not supported with float16 data type output");
    return false;
}

}
