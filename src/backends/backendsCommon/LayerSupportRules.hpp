//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/utility/Assert.hpp>
#include <algorithm>

namespace armnn
{

inline armnn::Optional<armnn::DataType> GetBiasTypeFromWeightsType(armnn::Optional<armnn::DataType> weightsType)
{
    if (!weightsType)
    {
        return weightsType;
    }

    switch(weightsType.value())
    {
        case armnn::DataType::Float16:
        case armnn::DataType::Float32:
            return weightsType;
        case armnn::DataType::QAsymmS8:
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::QSymmS8:
        case armnn::DataType::QSymmS16:
            return armnn::DataType::Signed32;
        default:
            ARMNN_ASSERT_MSG(false, "GetBiasTypeFromWeightsType(): Unsupported data type.");
    }
    return armnn::EmptyOptional();
}

template<typename F>
bool CheckSupportRule(F rule, Optional<std::string&> reasonIfUnsupported, const char* reason)
{
    bool supported = rule();
    if (!supported && reason)
    {
        reasonIfUnsupported.value() += std::string(reason) + "\n"; // Append the reason on a new line
    }
    return supported;
}

struct Rule
{
    bool operator()() const
    {
        return m_Res;
    }

    bool m_Res = true;
};

template<typename T>
bool AllTypesAreEqualImpl(T)
{
    return true;
}

template<typename T, typename... Rest>
bool AllTypesAreEqualImpl(T t1, T t2, Rest... rest)
{
    static_assert(std::is_same<T, TensorInfo>::value, "Type T must be a TensorInfo");

    return (t1.GetDataType() == t2.GetDataType()) && AllTypesAreEqualImpl(t2, rest...);
}

struct TypesAreEqual : public Rule
{
    template<typename ... Ts>
    TypesAreEqual(const Ts&... ts)
    {
        m_Res = AllTypesAreEqualImpl(ts...);
    }
};

struct QuantizationParametersAreEqual : public Rule
{
    QuantizationParametersAreEqual(const TensorInfo& info0, const TensorInfo& info1)
    {
        m_Res = info0.GetQuantizationScale() == info1.GetQuantizationScale() &&
                info0.GetQuantizationOffset() == info1.GetQuantizationOffset();
    }
};

struct TypeAnyOf : public Rule
{
    template<typename Container>
    TypeAnyOf(const TensorInfo& info, const Container& c)
    {
        m_Res = std::any_of(c.begin(), c.end(), [&info](DataType dt)
        {
            return dt == info.GetDataType();
        });
    }
};

struct TypeIs : public Rule
{
    TypeIs(const TensorInfo& info, DataType dt)
    {
        m_Res = dt == info.GetDataType();
    }
};

struct TypeNotPerAxisQuantized : public Rule
{
    TypeNotPerAxisQuantized(const TensorInfo& info)
    {
        m_Res = !info.IsQuantized() || !info.HasPerAxisQuantization();
    }
};

struct BiasAndWeightsTypesMatch : public Rule
{
    BiasAndWeightsTypesMatch(const TensorInfo& biases, const TensorInfo& weights)
    {
        m_Res = biases.GetDataType() == GetBiasTypeFromWeightsType(weights.GetDataType()).value();
    }
};

struct BiasAndWeightsTypesCompatible : public Rule
{
    template<typename Container>
    BiasAndWeightsTypesCompatible(const TensorInfo& info, const Container& c)
    {
        m_Res = std::any_of(c.begin(), c.end(), [&info](DataType dt)
            {
                return dt ==  GetBiasTypeFromWeightsType(info.GetDataType()).value();
            });
    }
};

struct ShapesAreSameRank : public Rule
{
    ShapesAreSameRank(const TensorInfo& info0, const TensorInfo& info1)
    {
        m_Res = info0.GetShape().GetNumDimensions() == info1.GetShape().GetNumDimensions();
    }
};

struct ShapesAreSameTotalSize : public Rule
{
    ShapesAreSameTotalSize(const TensorInfo& info0, const TensorInfo& info1)
    {
        m_Res = info0.GetNumElements() == info1.GetNumElements();
    }
};

struct ShapesAreBroadcastCompatible : public Rule
{
    unsigned int CalcInputSize(const TensorShape& in, const TensorShape& out, unsigned int idx)
    {
        unsigned int offset = out.GetNumDimensions() - in.GetNumDimensions();
        unsigned int sizeIn = (idx < offset) ? 1 : in[idx-offset];
        return sizeIn;
    }

    ShapesAreBroadcastCompatible(const TensorInfo& in0, const TensorInfo& in1, const TensorInfo& out)
    {
        const TensorShape& shape0 = in0.GetShape();
        const TensorShape& shape1 = in1.GetShape();
        const TensorShape& outShape = out.GetShape();

        for (unsigned int i=0; i < outShape.GetNumDimensions() && m_Res; i++)
        {
            unsigned int sizeOut = outShape[i];
            unsigned int sizeIn0 = CalcInputSize(shape0, outShape, i);
            unsigned int sizeIn1 = CalcInputSize(shape1, outShape, i);

            m_Res &= ((sizeIn0 == sizeOut) || (sizeIn0 == 1)) &&
                     ((sizeIn1 == sizeOut) || (sizeIn1 == 1));
        }
    }
};

struct TensorNumDimensionsAreCorrect : public Rule
{
    TensorNumDimensionsAreCorrect(const TensorInfo& info, unsigned int expectedNumDimensions)
    {
        m_Res = info.GetNumDimensions() == expectedNumDimensions;
    }
};

} //namespace armnn