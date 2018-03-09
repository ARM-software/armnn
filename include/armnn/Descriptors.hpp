//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "DescriptorsFwd.hpp"

#include <cstdint>
#include <initializer_list>

#include "Tensor.hpp"
#include "Types.hpp"

namespace armnn
{

struct ActivationDescriptor
{
    ActivationDescriptor() : m_Function(ActivationFunction::Sigmoid), m_A(0), m_B(0) {};

    ActivationFunction m_Function;
    float              m_A;
    float              m_B;
};

struct PermuteDescriptor
{
    PermuteDescriptor()
        : m_DimMappings{}
    {
    }
    PermuteDescriptor(const PermutationVector& dimMappings)
        : m_DimMappings(dimMappings)
    {
    }

    PermutationVector m_DimMappings;
};

struct SoftmaxDescriptor
{
    SoftmaxDescriptor() : m_Beta(1.0f) {};

    float              m_Beta;
};


struct OriginsDescriptor
{
    OriginsDescriptor();
    OriginsDescriptor(uint32_t numViews, uint32_t numDimensions = 4);
    OriginsDescriptor(const OriginsDescriptor& other);
    OriginsDescriptor(OriginsDescriptor&& other);

    ~OriginsDescriptor();

    OriginsDescriptor& operator=(OriginsDescriptor rhs);

    Status SetViewOriginCoord(uint32_t view, uint32_t coord, uint32_t value);
    uint32_t GetNumViews() const;
    uint32_t GetNumDimensions() const;
    const uint32_t* GetViewOrigin(uint32_t idx) const;
    void ReorderOrigins(unsigned int*  newOrdering, unsigned int numNewOrdering);
    friend void swap(OriginsDescriptor& first, OriginsDescriptor& second);

private:
    uint32_t   m_NumViews;
    uint32_t   m_NumDimensions;
    uint32_t** m_ViewOrigins;
};

struct ViewsDescriptor
{
    ViewsDescriptor(uint32_t numViews, uint32_t numDimensions = 4);
    ViewsDescriptor(const ViewsDescriptor& other);
    ViewsDescriptor();
    ViewsDescriptor(ViewsDescriptor&& other);

    ~ViewsDescriptor();

    ViewsDescriptor& operator=(ViewsDescriptor rhs);

    Status SetViewOriginCoord(uint32_t view, uint32_t coord, uint32_t value);
    Status SetViewSize(uint32_t view, uint32_t coord, uint32_t value);

    uint32_t GetNumViews() const;
    uint32_t GetNumDimensions() const;
    const uint32_t* GetViewOrigin(uint32_t idx) const;
    const uint32_t* GetViewSizes(uint32_t idx) const;

    friend void swap(ViewsDescriptor& first, ViewsDescriptor& second);
private:
    OriginsDescriptor m_Origins;
    uint32_t** m_ViewSizes;
};

// Convenience template to create a OriginsDescriptor to use when creating a Merger layer for performing concatenation
// of a number of input tensors
template <typename TensorShapeIt>
OriginsDescriptor CreateMergerDescriptorForConcatenation(TensorShapeIt first, TensorShapeIt last,
    unsigned int concatenationDimension)
{
    auto numInputs = std::distance(first, last);

    if (numInputs < 2)
    {
        throw InvalidArgumentException("Concatenation requires at least 2 inputs");
    }

    const auto& firstInputShape = *first;

    const unsigned int numDimensions = firstInputShape.GetNumDimensions();
    for (auto it = first + 1; it != last; ++it)
    {
        if (it->GetNumDimensions() != numDimensions)
        {
            throw InvalidArgumentException("All inputs to concatenation must have the same number of dimensions");
        }
    }

    if (concatenationDimension >= numDimensions)
    {
        throw InvalidArgumentException("concatenationDimension must be between 0 and the number of dimensions.");
    }

    for (auto it = first; it != last; ++it)
    {
        for (unsigned int d = 0; d < numDimensions; ++d)
        {
            const bool dimSizeOk = (d == concatenationDimension) || (firstInputShape[d] == (*it)[d]);
            if (!dimSizeOk)
            {
                throw InvalidArgumentException("All inputs to concatenation must be the same size along all dimensions "
                    " except the concatenation dimension");
            }
        }
    }

    OriginsDescriptor viewsDescriptor(static_cast<uint32_t>(numInputs), numDimensions);

    uint32_t viewIndex = 0u;
    uint32_t coordAlongConcatDim = 0u;
    for (auto it = first; it != last; ++it)
    {
        const auto& inputShape = *it;

        for (unsigned int i = 0; i < concatenationDimension; ++i)
        {
            viewsDescriptor.SetViewOriginCoord(viewIndex, i, 0);
        }

        viewsDescriptor.SetViewOriginCoord(viewIndex, concatenationDimension, coordAlongConcatDim);
        unsigned int dimSize = inputShape[concatenationDimension];
        coordAlongConcatDim += dimSize;


        for (unsigned int i = concatenationDimension + 1; i < numDimensions; ++i)
        {
            viewsDescriptor.SetViewOriginCoord(viewIndex, i, 0);
        }

        ++viewIndex;
    }

    return viewsDescriptor;
}

struct Pooling2dDescriptor
{
    Pooling2dDescriptor()
    : m_PoolType(PoolingAlgorithm::Max)
    , m_PadLeft(0)
    , m_PadRight(0)
    , m_PadTop(0)
    , m_PadBottom(0)
    , m_PoolWidth(0)
    , m_PoolHeight(0)
    , m_StrideX(0)
    , m_StrideY(0)
    , m_OutputShapeRounding(OutputShapeRounding::Floor)
    , m_PaddingMethod(PaddingMethod::Exclude)
    {};

    PoolingAlgorithm    m_PoolType;
    uint32_t            m_PadLeft;
    uint32_t            m_PadRight;
    uint32_t            m_PadTop;
    uint32_t            m_PadBottom;
    uint32_t            m_PoolWidth;
    uint32_t            m_PoolHeight;
    uint32_t            m_StrideX;
    uint32_t            m_StrideY;
    OutputShapeRounding m_OutputShapeRounding;
    PaddingMethod       m_PaddingMethod;
};

struct FullyConnectedDescriptor
{
    FullyConnectedDescriptor()
    : m_BiasEnabled(false)
    , m_TransposeWeightMatrix(false)
    {};

    bool m_BiasEnabled;
    bool m_TransposeWeightMatrix;
};

struct Convolution2dDescriptor
{
    Convolution2dDescriptor()
    : m_PadLeft(0)
    , m_PadRight(0)
    , m_PadTop(0)
    , m_PadBottom(0)
    , m_StrideX(0)
    , m_StrideY(0)
    , m_BiasEnabled(false)
    {};

    uint32_t             m_PadLeft;
    uint32_t             m_PadRight;
    uint32_t             m_PadTop;
    uint32_t             m_PadBottom;
    uint32_t             m_StrideX;
    uint32_t             m_StrideY;
    bool                 m_BiasEnabled;
};

struct DepthwiseConvolution2dDescriptor
{
    DepthwiseConvolution2dDescriptor()
    :   m_PadLeft(0)
    ,   m_PadRight(0)
    ,   m_PadTop(0)
    ,   m_PadBottom(0)
    ,   m_StrideX(0)
    ,   m_StrideY(0)
    ,   m_BiasEnabled(false)
    {}

    uint32_t m_PadLeft;
    uint32_t m_PadRight;
    uint32_t m_PadTop;
    uint32_t m_PadBottom;
    uint32_t m_StrideX;
    uint32_t m_StrideY;
    bool     m_BiasEnabled;
};


struct NormalizationDescriptor
{
    NormalizationDescriptor()
    : m_NormChannelType(NormalizationAlgorithmChannel::Across)
    , m_NormMethodType(NormalizationAlgorithmMethod::LocalBrightness)
    , m_NormSize(0)
    , m_Alpha(0.f)
    , m_Beta(0.f)
    , m_K(0.f)
    {}

    NormalizationAlgorithmChannel m_NormChannelType;
    NormalizationAlgorithmMethod  m_NormMethodType;
    uint32_t                      m_NormSize;
    float                         m_Alpha;
    float                         m_Beta;
    float                         m_K;
};

struct BatchNormalizationDescriptor
{
    BatchNormalizationDescriptor()
    : m_Eps(0.0001f)
    {}

    float m_Eps;
};

struct FakeQuantizationDescriptor
{
    FakeQuantizationDescriptor()
    : m_Min(-6.0f)
    , m_Max(6.0f)
    {}

    float m_Min;
    float m_Max;
};

struct ResizeBilinearDescriptor
{
    ResizeBilinearDescriptor()
    : m_TargetWidth(0)
    , m_TargetHeight(0)
    {}

    uint32_t m_TargetWidth;
    uint32_t m_TargetHeight;
};

struct ReshapeDescriptor
{
    TensorShape m_TargetShape;
};

}
