//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/TypesUtils.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnnUtils/FloatingPointConverter.hpp>
#include <armnnUtils/TensorUtils.hpp>

#include <ResolveType.hpp>

namespace armnn
{

class BaseIterator
{
public:
    BaseIterator() {}

    virtual ~BaseIterator() {}

    virtual BaseIterator& operator++() = 0;

    virtual BaseIterator& operator+=(const unsigned int increment) = 0;

    virtual BaseIterator& operator-=(const unsigned int increment) = 0;

    virtual BaseIterator& operator[](const unsigned int index) = 0;
};

template<typename IType>
class Decoder : public BaseIterator
{
public:
    Decoder() {}

    virtual ~Decoder() {}

    virtual void Reset(void*) = 0;

    virtual IType Get() const = 0;

    virtual std::vector<float>
    DecodeTensor(const TensorShape &tensorShape,
                 bool isDepthwise = false) = 0;
};

template<typename IType>
class Encoder : public BaseIterator
{
public:
    Encoder() {}

    virtual ~Encoder() {}

    virtual void Reset(void*) = 0;

    virtual void Set(IType right) = 0;

    virtual IType Get() const = 0;
};

template<typename T, typename Base>
class TypedIterator : public Base
{
public:
    TypedIterator(T* data = nullptr)
        : m_Iterator(data), m_Start(data)
    {}

    void Reset(void* data) override
    {
        m_Iterator = reinterpret_cast<T*>(data);
        m_Start = m_Iterator;
    }

    TypedIterator& operator++() override
    {
        ARMNN_ASSERT(m_Iterator);
        ++m_Iterator;
        return *this;
    }

    TypedIterator& operator+=(const unsigned int increment) override
    {
        ARMNN_ASSERT(m_Iterator);
        m_Iterator += increment;
        return *this;
    }

    TypedIterator& operator-=(const unsigned int increment) override
    {
        ARMNN_ASSERT(m_Iterator);
        m_Iterator -= increment;
        return *this;
    }

    TypedIterator& operator[](const unsigned int index) override
    {
        ARMNN_ASSERT(m_Iterator);
        m_Iterator = m_Start + index;
        return *this;
    }

protected:
    T* m_Iterator;
    T* m_Start;
};

class QASymm8Decoder : public TypedIterator<const uint8_t, Decoder<float>>
{
public:
    QASymm8Decoder(const uint8_t* data, const float scale, const int32_t offset)
        : TypedIterator(data), m_Scale(scale), m_Offset(offset) {}

    QASymm8Decoder(const float scale, const int32_t offset)
        : QASymm8Decoder(nullptr, scale, offset) {}

    float Get() const override
    {
        return armnn::Dequantize(*m_Iterator, m_Scale, m_Offset);
    }
    std::vector<float> DecodeTensor (const TensorShape& tensorShape,
                                     const bool isDepthwise) override
    {
        IgnoreUnused(isDepthwise);

        const unsigned int size = tensorShape.GetNumElements();
        std::vector<float> decodedTensor;
        decodedTensor.reserve(size);

        for (uint32_t i = 0; i < size; ++i)
        {
            this->operator[](i);
            decodedTensor.emplace_back(armnn::Dequantize(*m_Iterator, m_Scale, m_Offset));
        }

        return decodedTensor;
    }

private:

    const float m_Scale;
    const int32_t m_Offset;
};

class QASymmS8Decoder : public TypedIterator<const int8_t, Decoder<float>>
{
public:
    QASymmS8Decoder(const int8_t* data, const float scale, const int32_t offset)
        : TypedIterator(data), m_Scale(scale), m_Offset(offset) {}

    QASymmS8Decoder(const float scale, const int32_t offset)
        : QASymmS8Decoder(nullptr, scale, offset) {}

    float Get() const override
    {
        return armnn::Dequantize(*m_Iterator, m_Scale, m_Offset);
    }
    std::vector<float> DecodeTensor (const TensorShape& tensorShape,
                                     const bool isDepthwise) override
    {
        IgnoreUnused(isDepthwise);

        const unsigned int size = tensorShape.GetNumElements();
        std::vector<float> decodedTensor;
        decodedTensor.reserve(size);

        for (uint32_t i = 0; i < size; ++i)
        {
            this->operator[](i);
            decodedTensor.emplace_back(armnn::Dequantize(*m_Iterator, m_Scale, m_Offset));
        }

        return decodedTensor;
    }

private:
    const float m_Scale;
    const int32_t m_Offset;

};

class QSymmS8Decoder : public TypedIterator<const int8_t, Decoder<float>>
{
public:
    QSymmS8Decoder(const int8_t* data, const float scale, const int32_t offset)
            : TypedIterator(data), m_Scale(scale), m_Offset(offset) {}

    QSymmS8Decoder(const float scale, const int32_t offset)
            : QSymmS8Decoder(nullptr, scale, offset) {}

    float Get() const override
    {
        return armnn::Dequantize(*m_Iterator, m_Scale, m_Offset);
    }
    std::vector<float> DecodeTensor (const TensorShape& tensorShape,
                                     const bool isDepthwise) override
    {
        IgnoreUnused(isDepthwise);

        const unsigned int size = tensorShape.GetNumElements();
        std::vector<float> decodedTensor;
        decodedTensor.reserve(size);

        for (uint32_t i = 0; i < size; ++i)
        {
            this->operator[](i);
            decodedTensor.emplace_back(armnn::Dequantize(*m_Iterator, m_Scale, m_Offset));
        }

        return decodedTensor;
    }

private:
    const float m_Scale;
    const int32_t m_Offset;

};

class QSymm16Decoder : public TypedIterator<const int16_t, Decoder<float>>
{
public:
    QSymm16Decoder(const int16_t* data, const float scale, const int32_t offset)
        : TypedIterator(data), m_Scale(scale), m_Offset(offset) {}

    QSymm16Decoder(const float scale, const int32_t offset)
        : QSymm16Decoder(nullptr, scale, offset) {}

    float Get() const override
    {
        return armnn::Dequantize(*m_Iterator, m_Scale, m_Offset);
    }
    std::vector<float> DecodeTensor (const TensorShape& tensorShape,
                                     const bool isDepthwise) override
    {
        IgnoreUnused(isDepthwise);

        const unsigned int size = tensorShape.GetNumElements();
        std::vector<float> decodedTensor;
        decodedTensor.reserve(size);

        for (uint32_t i = 0; i < size; ++i)
        {
            this->operator[](i);
            decodedTensor.emplace_back(armnn::Dequantize(*m_Iterator, m_Scale, m_Offset));
        }

        return decodedTensor;
    }

private:
    const float m_Scale;
    const int32_t m_Offset;

};

class Float16Decoder : public TypedIterator<const Half, Decoder<float>>
{
public:
    Float16Decoder(const Half* data)
        : TypedIterator(data) {}

    Float16Decoder()
        : Float16Decoder(nullptr) {}

    float Get() const override
    {
        float val = 0.f;
        armnnUtils::FloatingPointConverter::ConvertFloat16To32(m_Iterator, 1, &val);
        return val;
    }
    std::vector<float> DecodeTensor (const TensorShape& tensorShape,
                                     const bool isDepthwise) override
    {
        IgnoreUnused(isDepthwise);

        const unsigned int size = tensorShape.GetNumElements();
        std::vector<float> decodedTensor;
        decodedTensor.reserve(size);

        for (uint32_t i = 0; i < size; ++i)
        {
            float val = 0.f;
            this->operator[](i);
            armnnUtils::FloatingPointConverter::ConvertFloat16To32(m_Iterator, 1, &val);
            decodedTensor.emplace_back(val);
        }

        return decodedTensor;
    }


};

class Float32Decoder : public TypedIterator<const float, Decoder<float>>
{
public:
    Float32Decoder(const float* data)
        : TypedIterator(data) {}

    Float32Decoder()
        : Float32Decoder(nullptr) {}

    float Get() const override
    {
        return *m_Iterator;
    }
    std::vector<float> DecodeTensor (const TensorShape& tensorShape,
                                     const bool isDepthwise) override
    {
        IgnoreUnused(isDepthwise);
        const unsigned int size = tensorShape.GetNumElements();
        std::vector<float> decodedTensor;

        decodedTensor.reserve(size);
        decodedTensor.assign(m_Start, m_Start + size);

        return decodedTensor;
    }
};

class ScaledInt32Decoder : public TypedIterator<const int32_t, Decoder<float>>
{
public:
    ScaledInt32Decoder(const int32_t* data, const float scale)
        : TypedIterator(data), m_Scale(scale) {}

    ScaledInt32Decoder(const float scale)
        : ScaledInt32Decoder(nullptr, scale) {}

    float Get() const override
    {
        return static_cast<float>(*m_Iterator) * m_Scale;
    }
    std::vector<float> DecodeTensor (const TensorShape& tensorShape,
                                     const bool isDepthwise) override
    {
        IgnoreUnused(isDepthwise);

        const unsigned int size = tensorShape.GetNumElements();
        std::vector<float> decodedTensor;
        decodedTensor.reserve(size);

        for (uint32_t i = 0; i < size; ++i)
        {
            this->operator[](i);
            decodedTensor.emplace_back(static_cast<float>(*m_Iterator) * m_Scale);
        }

        return decodedTensor;
    }

private:
    const float m_Scale;

};

class Int32Decoder : public TypedIterator<const int32_t, Decoder<float>>
{
public:
    Int32Decoder(const int32_t* data)
        : TypedIterator(data) {}

    Int32Decoder()
        : Int32Decoder(nullptr) {}

    float Get() const override
    {
        return static_cast<float>(*m_Iterator);
    }
    std::vector<float> DecodeTensor (const TensorShape& tensorShape,
                                     const bool isDepthwise) override
    {
        IgnoreUnused(isDepthwise);

        const unsigned int size = tensorShape.GetNumElements();
        std::vector<float> decodedTensor;
        decodedTensor.reserve(size);

        for (uint32_t i = 0; i < size; ++i)
        {
            this->operator[](i);
            decodedTensor.emplace_back(static_cast<float>(*m_Iterator));
        }

        return decodedTensor;
    }
};

class Int32ToInt32tDecoder : public TypedIterator<const int32_t, Decoder<int32_t>>
{
public:
    Int32ToInt32tDecoder(const int32_t* data)
            : TypedIterator(data){}

    Int32ToInt32tDecoder()
            : Int32ToInt32tDecoder(nullptr) {}

    int32_t Get() const override
    {
        return *m_Iterator;
    }
    std::vector<float> DecodeTensor (const TensorShape& tensorShape,
                                     const bool isDepthwise) override
    {
        IgnoreUnused(isDepthwise);

        const unsigned int size = tensorShape.GetNumElements();
        std::vector<float> decodedTensor;
        decodedTensor.reserve(size);

        for (uint32_t i = 0; i < size; ++i)
        {
            this->operator[](i);
            decodedTensor.emplace_back(static_cast<float>(*m_Iterator));
        }

        return decodedTensor;
    }
};

class BooleanDecoder : public TypedIterator<const uint8_t, Decoder<float>>
{
public:
    BooleanDecoder(const uint8_t* data)
            : TypedIterator(data) {}

    BooleanDecoder()
            : BooleanDecoder(nullptr) {}

    float Get() const override
    {
        return *m_Iterator;
    }
    std::vector<float> DecodeTensor (const TensorShape& tensorShape,
                                     const bool isDepthwise) override
    {
        IgnoreUnused(isDepthwise);

        const unsigned int size = tensorShape.GetNumElements();
        std::vector<float> decodedTensor;
        decodedTensor.reserve(size);

        for (uint32_t i = 0; i < size; ++i)
        {
            this->operator[](i);
            decodedTensor.emplace_back(*m_Iterator);
        }

        return decodedTensor;
    }
};

class BooleanDecoderBool : public TypedIterator<const uint8_t, Decoder<bool>>
{
public:
    BooleanDecoderBool(const uint8_t* data)
        : TypedIterator(data) {}

    BooleanDecoderBool()
        : BooleanDecoderBool(nullptr) {}

    bool Get() const override
    {
        return *m_Iterator;
    }

    std::vector<float> DecodeTensor(const TensorShape& tensorShape,
                                    const bool isDepthwise) override
    {
        IgnoreUnused(isDepthwise);

        const unsigned int size = tensorShape.GetNumElements();
        std::vector<float> decodedTensor;
        decodedTensor.reserve(size);

        for (uint32_t i = 0; i < size; ++i)
        {
            this->operator[](i);
            decodedTensor.emplace_back(*m_Iterator);
        }

        return decodedTensor;
    }
};

class QASymm8Encoder : public TypedIterator<uint8_t, Encoder<float>>
{
public:
    QASymm8Encoder(uint8_t* data, const float scale, const int32_t offset)
        : TypedIterator(data), m_Scale(scale), m_Offset(offset) {}

    QASymm8Encoder(const float scale, const int32_t offset)
        : QASymm8Encoder(nullptr, scale, offset) {}

    void Set(float right) override
    {
        *m_Iterator = armnn::Quantize<uint8_t>(right, m_Scale, m_Offset);
    }

    float Get() const override
    {
        return armnn::Dequantize(*m_Iterator, m_Scale, m_Offset);
    }

private:
    const float m_Scale;
    const int32_t m_Offset;
};

class QASymmS8Encoder : public TypedIterator<int8_t, Encoder<float>>
{
public:
    QASymmS8Encoder(int8_t* data, const float scale, const int32_t offset)
        : TypedIterator(data), m_Scale(scale), m_Offset(offset) {}

    QASymmS8Encoder(const float scale, const int32_t offset)
        : QASymmS8Encoder(nullptr, scale, offset) {}

    void Set(float right) override
    {
        *m_Iterator = armnn::Quantize<int8_t>(right, m_Scale, m_Offset);
    }

    float Get() const override
    {
        return armnn::Dequantize(*m_Iterator, m_Scale, m_Offset);
    }

private:
    const float m_Scale;
    const int32_t m_Offset;
};

class QSymmS8Encoder : public TypedIterator<int8_t, Encoder<float>>
{
public:
    QSymmS8Encoder(int8_t* data, const float scale, const int32_t offset)
            : TypedIterator(data), m_Scale(scale), m_Offset(offset) {}

    QSymmS8Encoder(const float scale, const int32_t offset)
            : QSymmS8Encoder(nullptr, scale, offset) {}

    void Set(float right) override
    {
        *m_Iterator = armnn::Quantize<int8_t>(right, m_Scale, m_Offset);
    }

    float Get() const override
    {
        return armnn::Dequantize(*m_Iterator, m_Scale, m_Offset);
    }

private:
    const float m_Scale;
    const int32_t m_Offset;
};

class QSymm16Encoder : public TypedIterator<int16_t, Encoder<float>>
{
public:
    QSymm16Encoder(int16_t* data, const float scale, const int32_t offset)
        : TypedIterator(data), m_Scale(scale), m_Offset(offset) {}

    QSymm16Encoder(const float scale, const int32_t offset)
        : QSymm16Encoder(nullptr, scale, offset) {}

    void Set(float right) override
    {
        *m_Iterator = armnn::Quantize<int16_t>(right, m_Scale, m_Offset);
    }

    float Get() const override
    {
        return armnn::Dequantize(*m_Iterator, m_Scale, m_Offset);
    }

private:
    const float m_Scale;
    const int32_t m_Offset;
};

class Float16Encoder : public TypedIterator<Half, Encoder<float>>
{
public:
    Float16Encoder(Half* data)
        : TypedIterator(data) {}

    Float16Encoder()
        : Float16Encoder(nullptr) {}

    void Set(float right) override
    {
        armnnUtils::FloatingPointConverter::ConvertFloat32To16(&right, 1, m_Iterator);
    }

    float Get() const override
    {
        float val = 0.f;
        armnnUtils::FloatingPointConverter::ConvertFloat16To32(m_Iterator, 1, &val);
        return val;
    }
};

class Float32Encoder : public TypedIterator<float, Encoder<float>>
{
public:
    Float32Encoder(float* data)
        : TypedIterator(data) {}

    Float32Encoder()
        : Float32Encoder(nullptr) {}

    void Set(float right) override
    {
        *m_Iterator = right;
    }

    float Get() const override
    {
        return *m_Iterator;
    }
};

class Int32Encoder : public TypedIterator<int32_t, Encoder<float>>
{
public:
    Int32Encoder(int32_t* data)
        : TypedIterator(data) {}

    Int32Encoder()
        : Int32Encoder(nullptr) {}

    void Set(float right) override
    {
        *m_Iterator = static_cast<int32_t>(right);
    }

    float Get() const override
    {
        return static_cast<float>(*m_Iterator);
    }
};

class Int32ToInt32tEncoder : public TypedIterator<int32_t, Encoder<int32_t>>
{
public:
    Int32ToInt32tEncoder(int32_t* data)
        : TypedIterator(data){}

    Int32ToInt32tEncoder()
        : Int32ToInt32tEncoder(nullptr) {}

    void Set(int32_t right) override
    {
        *m_Iterator = right;
    }

    int32_t Get() const override
    {
        return *m_Iterator;
    }
};

class BooleanEncoder : public TypedIterator<uint8_t, Encoder<bool>>
{
public:
    BooleanEncoder(uint8_t* data)
        : TypedIterator(data) {}

    BooleanEncoder()
        : BooleanEncoder(nullptr) {}

    void Set(bool right) override
    {
        *m_Iterator = right;
    }

    bool Get() const override
    {
        return *m_Iterator;
    }
};

/// PerAxisIterator for per-axis quantization. Iterates over a tensor as layed out in memory and keeps track
/// of the axis index.
template<typename T, typename Base>
class PerAxisIterator : public Base
{
public:
    PerAxisIterator(T* data = nullptr,
                    unsigned int axisFactor = 0,
                    unsigned int axisDimensionality=0)
        : m_Iterator(data),
          m_Start(data),
          m_AxisIndex(0), // iterates over the dimension of axis
          m_AxisDimensionality(axisDimensionality), // tensorShape[quantization_dim]
          m_AxisFactor(axisFactor),
          m_Index(0)
    {}

    PerAxisIterator(T* data = nullptr,
                    const armnn::TensorShape& tensorShape = TensorShape(),
                    const unsigned int axis = 0)
        : m_Iterator(data),
          m_Start(data),
          m_AxisIndex(0),
          m_Index(0)
    {
        m_AxisDimensionality = tensorShape[axis];
        m_AxisFactor = armnnUtils::GetNumElementsAfter(tensorShape, axis);
    }

    void Reset(void* data) override
    {
        m_Iterator = reinterpret_cast<T*>(data);
        m_Start = m_Iterator;
        m_AxisIndex = 0;
        m_Index = 0;
    }

    PerAxisIterator& operator++() override
    {
        ++m_Index;
        this -> operator[](m_Index);
        return *this;
    }

    PerAxisIterator& operator+=(const unsigned int increment) override
    {
        m_Index += increment;
        this -> operator[](m_Index);
        return *this;
    }

    PerAxisIterator& operator-=(const unsigned int decrement) override
    {
        m_Index -= decrement;
        this -> operator[](m_Index);
        return *this;
    }


    inline PerAxisIterator& SetIndexOnMem(const unsigned int index)
    {
        ARMNN_ASSERT(m_Iterator);
        m_Iterator = m_Start + index;
        if (index < m_AxisFactor)
        {
            m_AxisIndex = 0;
        }
        else
        {
            m_AxisIndex = (index / m_AxisFactor) % m_AxisDimensionality;
        }
        m_Index = index;
        return *this;
    }

    PerAxisIterator& operator[](const unsigned int index) override
    {
        SetIndexOnMem(index);
        return *this;
    }

    protected:
        T* m_Iterator;
        T* m_Start;
        unsigned int m_AxisIndex;
        unsigned int m_AxisDimensionality; // tensorShape[quantization_dim]
        unsigned int m_AxisFactor;
        unsigned int m_Index;
};

class QSymm8PerAxisDecoder : public PerAxisIterator<const int8_t, Decoder<float>>
{
public:
    QSymm8PerAxisDecoder(const int8_t* data, const armnn::TensorInfo& tensorInfo)
            : PerAxisIterator(data, tensorInfo.GetShape(), tensorInfo.GetQuantizationDim().value()),
              m_Scales(tensorInfo.GetQuantizationScales())
    {}

    float Get() const override
    {
        return armnn::Dequantize(*m_Iterator, GetScale(), 0);
    }

    // Get scale of the current value
    float GetScale() const
    {
        return m_Scales[m_AxisIndex];
    }

    std::vector<float> DecodeTensor(const TensorShape &tensorShape,
                                    bool isDepthwise) override
    {
        IgnoreUnused(isDepthwise);

        const unsigned int size = tensorShape.GetNumElements();
        std::vector<float> decodedTensor;
        decodedTensor.reserve(size);

        for (uint32_t i = 0; i < size; ++i)
        {
            SetIndexOnMem(i);
            decodedTensor.emplace_back(armnn::Dequantize(*m_Iterator, GetScale(), 0));
        }
        return decodedTensor;
    }

private:
    std::vector<float> m_Scales;
};

class QSymm8PerAxisEncoder : public PerAxisIterator<int8_t, Encoder<float>>
{
public:
    QSymm8PerAxisEncoder(int8_t* data, const std::vector<float>& scale, unsigned int axisFactor)
        : PerAxisIterator(data, axisFactor), m_Scale(scale) {}

    void Set(float right)
    {
        *m_Iterator = armnn::Quantize<int8_t>(right, m_Scale[m_AxisIndex], 0);
    }

    float Get() const
    {
        return armnn::Dequantize(*m_Iterator, m_Scale[m_AxisIndex], 0);
    }

    // Get scale of the current value
    float GetScale() const
    {
        return m_Scale[m_AxisIndex];
    }

private:
    std::vector<float> m_Scale;
};

class ScaledInt32PerAxisDecoder : public PerAxisIterator<const int32_t, Decoder<float>>
{
public:
    ScaledInt32PerAxisDecoder(const int32_t* data, const armnn::TensorInfo tensorInfo)
    : PerAxisIterator(data, tensorInfo.GetShape(), tensorInfo.GetQuantizationDim().value()),
      m_Scales(tensorInfo.GetQuantizationScales())
    {}

    float Get() const override
    {
        return armnn::Dequantize(*m_Iterator, m_Scales[m_AxisIndex], 0);
    }

    // Get scale of the current value
    float GetScale() const
    {
        return m_Scales[m_AxisIndex];
    }

    std::vector<float> DecodeTensor(const TensorShape &tensorShape,
                                    bool isDepthwise) override
    {
        const uint32_t size = tensorShape.GetNumElements();

        const uint32_t stepSize = isDepthwise ?
                                  tensorShape[2] * tensorShape[3] : tensorShape.GetNumElements() / tensorShape[0];

        const uint32_t stepNum = size / stepSize;

        std::vector<float> decodedTensor;
        decodedTensor.reserve(size);

        // channelMultiplier is only used in depthwise convolutions and in other cases will have no effect
        // stepSize is the length of a contiguous area sharing a quantization scale within a tensor
        // stepNum is the number of those steps/blocks in the tensor
        for (uint32_t step = 0; step < stepNum; ++step)
        {
            //scale = (channelMultiplier * step + mult) % scaleSize;
            for (uint32_t i = 0; i < stepSize; ++i)
            {
                unsigned int index = step * stepSize + i;
                this->operator[](index);
                decodedTensor.emplace_back(armnn::Dequantize(*m_Iterator, m_Scales[step], 0));
            }
        }
        return decodedTensor;
    }

private:
    std::vector<float> m_Scales;
};

} // namespace armnn
