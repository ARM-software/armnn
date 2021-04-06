//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Deprecated.hpp"
#include "DescriptorsFwd.hpp"

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <sstream>

#include "Tensor.hpp"
#include "Types.hpp"

namespace armnn
{

/// Base class for all descriptors.
struct BaseDescriptor {};

/// An ActivationDescriptor for the ActivationLayer.
struct ActivationDescriptor : BaseDescriptor
{
    ActivationDescriptor()
        : m_Function(ActivationFunction::Sigmoid)
        , m_A(0)
        , m_B(0)
    {}

    ActivationDescriptor(armnn::ActivationFunction activation,
                         float a = 0,
                         float b = 0)
            : m_Function(activation)
            , m_A(a)
            , m_B(b)
    {}

    bool operator ==(const ActivationDescriptor &rhs) const
    {
        return m_Function == rhs.m_Function && m_A == rhs.m_B && m_B == rhs.m_B;
    }

    /// @brief The activation function to use
    /// (Sigmoid, TanH, Linear, ReLu, BoundedReLu, SoftReLu, LeakyReLu, Abs, Sqrt, Square, Elu).
    ActivationFunction m_Function;
    /// Alpha upper bound value used by the activation functions. (BoundedReLu, Linear, TanH, Elu).
    float              m_A;
    /// Beta lower bound value used by the activation functions. (BoundedReLu, Linear, TanH).
    float              m_B;
};

/// An ArgMinMaxDescriptor for ArgMinMaxLayer
struct ArgMinMaxDescriptor : BaseDescriptor
{
    ArgMinMaxDescriptor()
        : m_Function(ArgMinMaxFunction::Min)
        , m_Axis(-1)
        , m_Output_Type(armnn::DataType::Signed32)
    {}

    bool operator ==(const ArgMinMaxDescriptor &rhs) const
    {
        return m_Function == rhs.m_Function && m_Axis == rhs.m_Axis && m_Output_Type == rhs.m_Output_Type;
    }

    /// Specify if the function is to find Min or Max.
    ArgMinMaxFunction m_Function;
    /// Axis to reduce across the input tensor.
    int m_Axis;
    /// Deprecated and will be removed in future release.
    armnn::DataType m_Output_Type;
};

/// A ComparisonDescriptor for the ComparisonLayer
struct ComparisonDescriptor : BaseDescriptor
{
    ComparisonDescriptor()
        : ComparisonDescriptor(ComparisonOperation::Equal)
    {}

    ComparisonDescriptor(ComparisonOperation operation)
        : m_Operation(operation)
    {}

    bool operator ==(const ComparisonDescriptor &rhs) const
    {
        return m_Operation == rhs.m_Operation;
    }

    /// Specifies the comparison operation to execute
    ComparisonOperation m_Operation;
};

/// A ElementwiseUnaryDescriptor for the ElementwiseUnaryLayer
struct ElementwiseUnaryDescriptor : BaseDescriptor
{
    ElementwiseUnaryDescriptor()
        : ElementwiseUnaryDescriptor(UnaryOperation::Abs)
    {}

    ElementwiseUnaryDescriptor(UnaryOperation operation)
        : m_Operation(operation)
    {}

    bool operator ==(const ElementwiseUnaryDescriptor &rhs) const
    {
        return m_Operation == rhs.m_Operation;
    }

    /// Specifies the elementwiseUnary operation to execute
    UnaryOperation m_Operation;
};

/// A PermuteDescriptor for the PermuteLayer.
struct PermuteDescriptor : BaseDescriptor
{
    PermuteDescriptor()
        : m_DimMappings{}
    {}

    PermuteDescriptor(const PermutationVector& dimMappings)
        : m_DimMappings(dimMappings)
    {}

    bool operator ==(const PermuteDescriptor &rhs) const
    {
        return m_DimMappings.IsEqual(rhs.m_DimMappings);
    }

    /// @brief Indicates how to translate tensor elements from a given source into the target destination, when
    /// source and target potentially have different memory layouts e.g. {0U, 3U, 1U, 2U}.
    PermutationVector m_DimMappings;
};

/// A SoftmaxDescriptor for the SoftmaxLayer.
struct SoftmaxDescriptor : BaseDescriptor
{
    SoftmaxDescriptor()
        : m_Beta(1.0f)
        , m_Axis(-1)
    {}

    bool operator ==(const SoftmaxDescriptor& rhs) const
    {
        return m_Beta == rhs.m_Beta && m_Axis == rhs.m_Axis;
    }

    /// Exponentiation value.
    float m_Beta;
    /// Scalar, defaulted to the last index (-1), specifying the dimension the activation will be performed on.
    int m_Axis;
};

/// A LogSoftmaxDescriptor for the LogSoftmaxLayer
using LogSoftmaxDescriptor = SoftmaxDescriptor;

/// @brief An OriginsDescriptor for the ConcatLayer.
/// Descriptor to configure the concatenation process. Number of views must be equal to the number of inputs, and
/// their order must match - e.g. first view corresponds to the first input, second view to the second input, etc.
struct OriginsDescriptor : BaseDescriptor
{
    OriginsDescriptor();
    OriginsDescriptor(uint32_t numViews, uint32_t numDimensions = 4);
    OriginsDescriptor(const OriginsDescriptor& other);
    OriginsDescriptor(OriginsDescriptor&& other);

    ~OriginsDescriptor();

    OriginsDescriptor& operator=(OriginsDescriptor rhs);

    bool operator ==(const OriginsDescriptor& rhs) const;

    /// @Brief Set the view origin coordinates. The arguments are: view, dimension, value.
    /// If the view is greater than or equal to GetNumViews(), then the view argument is out of range.
    /// If the coord is greater than or equal to GetNumDimensions(), then the coord argument is out of range.
    Status SetViewOriginCoord(uint32_t view, uint32_t coord, uint32_t value);
    /// Get the number of views.
    uint32_t GetNumViews() const;
    /// Get the number of dimensions.
    uint32_t GetNumDimensions() const;
    /// Return the view origin at the int value idx.
    const uint32_t* GetViewOrigin(uint32_t idx) const;
    /// @brief Reorders the viewOrigins in accordance with the indices presented in newOrdering array.
    /// The number of views must match number of elements in the new ordering array.
    void ReorderOrigins(unsigned int*  newOrdering, unsigned int numNewOrdering);
    /// Swap the ViewsDescriptor values first and second.
    friend void swap(OriginsDescriptor& first, OriginsDescriptor& second);
    /// Set the concatenation axis value.
    void SetConcatAxis(unsigned int concatAxis);
    /// Get the concatenation axis value.
    unsigned int GetConcatAxis() const;

private:
    unsigned int m_ConcatAxis;
    uint32_t     m_NumViews;
    uint32_t     m_NumDimensions;
    uint32_t**   m_ViewOrigins;
};

/// @brief A ViewsDescriptor for the SplitterLayer.
/// Descriptor to configure the splitting process. Number of Views must be equal to the number of outputs, and
/// their order must match - e.g. first view corresponds to the first output, second view to the second output, etc.
struct ViewsDescriptor : BaseDescriptor
{
    ViewsDescriptor(uint32_t numViews, uint32_t numDimensions = 4);
    ViewsDescriptor(const ViewsDescriptor& other);
    ViewsDescriptor();
    ViewsDescriptor(ViewsDescriptor&& other);

    ~ViewsDescriptor();

    ViewsDescriptor& operator=(ViewsDescriptor rhs);

    bool operator ==(const ViewsDescriptor& rhs) const;

    /// @Brief Set the view origin coordinates. The arguments are: view, dimension, value.
    /// If the view is greater than or equal to GetNumViews(), then the view argument is out of range.
    /// If the coord is greater than or equal to GetNumDimensions(), then the coord argument is out of range.
    Status SetViewOriginCoord(uint32_t view, uint32_t coord, uint32_t value);
    /// @brief Set the size of the views. The arguments are: view, dimension, value.
    /// If the view is greater than or equal to GetNumViews(), then the view argument is out of range.
    /// If the coord is greater than or equal to GetNumDimensions(), then the coord argument is out of range.
    Status SetViewSize(uint32_t view, uint32_t coord, uint32_t value);

    /// Get the number of views.
    uint32_t GetNumViews() const;
    /// Get the number of dimensions.
    uint32_t GetNumDimensions() const;
    /// Get the view origin at the int value idx.
    const uint32_t* GetViewOrigin(uint32_t idx) const;
    /// Get the view sizes at the int value idx.
    const uint32_t* GetViewSizes(uint32_t idx) const;
    /// Get the View Origins
    const OriginsDescriptor& GetOrigins() const;

    /// Swap the ViewsDescriptor value first and second.
    friend void swap(ViewsDescriptor& first, ViewsDescriptor& second);
private:
    OriginsDescriptor m_Origins;
    uint32_t**        m_ViewSizes;
};

template <typename TensorShapeIt>
ARMNN_DEPRECATED_MSG("Use CreateDescriptorForConcatenation instead")
OriginsDescriptor CreateMergerDescriptorForConcatenation(TensorShapeIt first,
                                                         TensorShapeIt last,
                                                         unsigned int concatenationDimension)
{
    return CreateDescriptorForConcatenation(first, last, concatenationDimension);
}

/// @brief Convenience template to create an OriginsDescriptor to use when creating a ConcatLayer for performing
/// concatenation of a number of input tensors.
template <typename TensorShapeIt>
OriginsDescriptor CreateDescriptorForConcatenation(TensorShapeIt first,
                                                   TensorShapeIt last,
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
    viewsDescriptor.SetConcatAxis(concatenationDimension);

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

/// A Pooling2dDescriptor for the Pooling2dLayer.
struct Pooling2dDescriptor : BaseDescriptor
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
        , m_DataLayout(DataLayout::NCHW)
    {}

    bool operator ==(const Pooling2dDescriptor& rhs) const
    {
        return m_PoolType            == rhs.m_PoolType &&
               m_PadLeft             == rhs.m_PadLeft &&
               m_PadRight            == rhs.m_PadRight &&
               m_PadTop              == rhs.m_PadTop &&
               m_PadBottom           == rhs.m_PadBottom &&
               m_PoolWidth           == rhs.m_PoolWidth &&
               m_PoolHeight          == rhs.m_PoolHeight &&
               m_StrideX             == rhs.m_StrideX &&
               m_StrideY             == rhs.m_StrideY &&
               m_OutputShapeRounding == rhs.m_OutputShapeRounding &&
               m_PaddingMethod       == rhs.m_PaddingMethod &&
               m_DataLayout          == rhs.m_DataLayout;
    }

    /// The pooling algorithm to use (Max. Average, L2).
    PoolingAlgorithm    m_PoolType;
    /// Padding left value in the width dimension.
    uint32_t            m_PadLeft;
    /// Padding right value in the width dimension.
    uint32_t            m_PadRight;
    /// Padding top value in the height dimension.
    uint32_t            m_PadTop;
    /// Padding bottom value in the height dimension.
    uint32_t            m_PadBottom;
    /// Pooling width value.
    uint32_t            m_PoolWidth;
    /// Pooling height value.
    uint32_t            m_PoolHeight;
    /// Stride value when proceeding through input for the width dimension.
    uint32_t            m_StrideX;
    /// Stride value when proceeding through input for the height dimension.
    uint32_t            m_StrideY;
    /// The rounding method for the output shape. (Floor, Ceiling).
    OutputShapeRounding m_OutputShapeRounding;
    /// The padding method to be used. (Exclude, IgnoreValue).
    PaddingMethod       m_PaddingMethod;
    /// The data layout to be used (NCHW, NHWC).
    DataLayout   m_DataLayout;
};

/// A FullyConnectedDescriptor for the FullyConnectedLayer.
struct FullyConnectedDescriptor : BaseDescriptor
{
    FullyConnectedDescriptor()
        : m_BiasEnabled(false)
        , m_TransposeWeightMatrix(false)
        , m_ConstantWeights(true)
    {}

    bool operator ==(const FullyConnectedDescriptor& rhs) const
    {
        return m_BiasEnabled == rhs.m_BiasEnabled
               && m_TransposeWeightMatrix == rhs.m_TransposeWeightMatrix
               && m_ConstantWeights == rhs.m_ConstantWeights;
    }

    /// Get the number of views/inputs.
    uint32_t GetNumViews() const;

    /// Enable/disable bias.
    bool m_BiasEnabled;
    /// Enable/disable transpose weight matrix.
    bool m_TransposeWeightMatrix;
    /// Enable/disable constant weights and biases.
    bool m_ConstantWeights;
};

/// A Convolution2dDescriptor for the Convolution2dLayer.
struct Convolution2dDescriptor : BaseDescriptor
{
    Convolution2dDescriptor()
        : m_PadLeft(0)
        , m_PadRight(0)
        , m_PadTop(0)
        , m_PadBottom(0)
        , m_StrideX(1)
        , m_StrideY(1)
        , m_DilationX(1)
        , m_DilationY(1)
        , m_BiasEnabled(false)
        , m_DataLayout(DataLayout::NCHW)
    {}

    bool operator ==(const Convolution2dDescriptor& rhs) const
    {
        return m_PadLeft     == rhs.m_PadLeft &&
               m_PadRight    == rhs.m_PadRight &&
               m_PadTop      == rhs.m_PadTop &&
               m_PadBottom   == rhs.m_PadBottom &&
               m_StrideX     == rhs.m_StrideX &&
               m_StrideY     == rhs.m_StrideY &&
               m_DilationX   == rhs.m_DilationX &&
               m_DilationY   == rhs.m_DilationY &&
               m_BiasEnabled == rhs.m_BiasEnabled &&
               m_DataLayout  == rhs.m_DataLayout;
    }

    /// Padding left value in the width dimension.
    uint32_t             m_PadLeft;
    /// Padding right value in the width dimension.
    uint32_t             m_PadRight;
    /// Padding top value in the height dimension.
    uint32_t             m_PadTop;
    /// Padding bottom value in the height dimension.
    uint32_t             m_PadBottom;
    /// Stride value when proceeding through input for the width dimension.
    uint32_t             m_StrideX;
    /// Stride value when proceeding through input for the height dimension.
    uint32_t             m_StrideY;
    /// Dilation along x axis
    uint32_t             m_DilationX;
    /// Dilation along y axis
    uint32_t             m_DilationY;
    /// Enable/disable bias.
    bool                 m_BiasEnabled;
    /// The data layout to be used (NCHW, NHWC).
    DataLayout           m_DataLayout;
};

/// A DepthwiseConvolution2dDescriptor for the DepthwiseConvolution2dLayer.
struct DepthwiseConvolution2dDescriptor : BaseDescriptor
{
    DepthwiseConvolution2dDescriptor()
        : m_PadLeft(0)
        , m_PadRight(0)
        , m_PadTop(0)
        , m_PadBottom(0)
        , m_StrideX(1)
        , m_StrideY(1)
        , m_DilationX(1)
        , m_DilationY(1)
        , m_BiasEnabled(false)
        , m_DataLayout(DataLayout::NCHW)
    {}

    bool operator ==(const DepthwiseConvolution2dDescriptor& rhs) const
    {
        return m_PadLeft     == rhs.m_PadLeft &&
               m_PadRight    == rhs.m_PadRight &&
               m_PadTop      == rhs.m_PadTop &&
               m_PadBottom   == rhs.m_PadBottom &&
               m_StrideX     == rhs.m_StrideX &&
               m_StrideY     == rhs.m_StrideY &&
               m_DilationX   == rhs.m_DilationX &&
               m_DilationY   == rhs.m_DilationY &&
               m_BiasEnabled == rhs.m_BiasEnabled &&
               m_DataLayout  == rhs.m_DataLayout;
    }

    /// Padding left value in the width dimension.
    uint32_t   m_PadLeft;
    /// Padding right value in the width dimension.
    uint32_t   m_PadRight;
    /// Padding top value in the height dimension.
    uint32_t   m_PadTop;
    /// Padding bottom value in the height dimension.
    uint32_t   m_PadBottom;
    /// Stride value when proceeding through input for the width dimension.
    uint32_t   m_StrideX;
    /// Stride value when proceeding through input for the height dimension.
    uint32_t   m_StrideY;
    /// Dilation factor value for width dimension.
    uint32_t   m_DilationX;
    /// Dilation factor value for height dimension.
    uint32_t   m_DilationY;
    /// Enable/disable bias.
    bool       m_BiasEnabled;
    /// The data layout to be used (NCHW, NHWC).
    DataLayout m_DataLayout;
};

struct DetectionPostProcessDescriptor : BaseDescriptor
{
    DetectionPostProcessDescriptor()
        : m_MaxDetections(0)
        , m_MaxClassesPerDetection(1)
        , m_DetectionsPerClass(1)
        , m_NmsScoreThreshold(0)
        , m_NmsIouThreshold(0)
        , m_NumClasses(0)
        , m_UseRegularNms(false)
        , m_ScaleX(0)
        , m_ScaleY(0)
        , m_ScaleW(0)
        , m_ScaleH(0)
    {}

    bool operator ==(const DetectionPostProcessDescriptor& rhs) const
    {
        return m_MaxDetections          == rhs.m_MaxDetections &&
               m_MaxClassesPerDetection == rhs.m_MaxClassesPerDetection &&
               m_DetectionsPerClass     == rhs.m_DetectionsPerClass &&
               m_NmsScoreThreshold      == rhs.m_NmsScoreThreshold &&
               m_NmsIouThreshold        == rhs.m_NmsIouThreshold &&
               m_NumClasses             == rhs.m_NumClasses &&
               m_UseRegularNms          == rhs.m_UseRegularNms &&
               m_ScaleX                 == rhs.m_ScaleX &&
               m_ScaleY                 == rhs.m_ScaleY &&
               m_ScaleW                 == rhs.m_ScaleW &&
               m_ScaleH                 == rhs.m_ScaleH;
    }

    /// Maximum numbers of detections.
    uint32_t m_MaxDetections;
    /// Maximum numbers of classes per detection, used in Fast NMS.
    uint32_t m_MaxClassesPerDetection;
    /// Detections per classes, used in Regular NMS.
    uint32_t m_DetectionsPerClass;
    /// NMS score threshold.
    float m_NmsScoreThreshold;
    /// Intersection over union threshold.
    float m_NmsIouThreshold;
    /// Number of classes.
    uint32_t m_NumClasses;
    /// Use Regular NMS.
    bool m_UseRegularNms;
    /// Center size encoding scale x.
    float m_ScaleX;
    /// Center size encoding scale y.
    float m_ScaleY;
    /// Center size encoding scale weight.
    float m_ScaleW;
    /// Center size encoding scale height.
    float m_ScaleH;
};

/// A NormalizationDescriptor for the NormalizationLayer.
struct NormalizationDescriptor : BaseDescriptor
{
    NormalizationDescriptor()
        : m_NormChannelType(NormalizationAlgorithmChannel::Across)
        , m_NormMethodType(NormalizationAlgorithmMethod::LocalBrightness)
        , m_NormSize(0)
        , m_Alpha(0.f)
        , m_Beta(0.f)
        , m_K(0.f)
        , m_DataLayout(DataLayout::NCHW)
    {}

    bool operator ==(const NormalizationDescriptor& rhs) const
    {
        return m_NormChannelType == rhs.m_NormChannelType &&
               m_NormMethodType  == rhs.m_NormMethodType &&
               m_NormSize        == rhs.m_NormSize &&
               m_Alpha           == rhs.m_Alpha &&
               m_Beta            == rhs.m_Beta &&
               m_K               == rhs.m_K &&
               m_DataLayout      == rhs.m_DataLayout;
    }

    /// Normalization channel algorithm to use (Across, Within).
    NormalizationAlgorithmChannel m_NormChannelType;
    /// Normalization method algorithm to use (LocalBrightness, LocalContrast).
    NormalizationAlgorithmMethod  m_NormMethodType;
    /// Depth radius value.
    uint32_t                      m_NormSize;
    /// Alpha value for the normalization equation.
    float                         m_Alpha;
    /// Beta value for the normalization equation.
    float                         m_Beta;
    /// Kappa value used for the across channel normalization equation.
    float                         m_K;
    /// The data layout to be used (NCHW, NHWC).
    DataLayout                    m_DataLayout;
};

/// A L2NormalizationDescriptor for the L2NormalizationLayer.
struct L2NormalizationDescriptor : BaseDescriptor
{
    L2NormalizationDescriptor()
        : m_Eps(1e-12f)
        , m_DataLayout(DataLayout::NCHW)
    {}

    bool operator ==(const L2NormalizationDescriptor& rhs) const
    {
        return m_Eps == rhs.m_Eps && m_DataLayout == rhs.m_DataLayout;
    }

    /// Used to avoid dividing by zero.
    float m_Eps;
    /// The data layout to be used (NCHW, NHWC).
    DataLayout m_DataLayout;
};

/// A BatchNormalizationDescriptor for the BatchNormalizationLayer.
struct BatchNormalizationDescriptor : BaseDescriptor
{
    BatchNormalizationDescriptor()
        : m_Eps(0.0001f)
        , m_DataLayout(DataLayout::NCHW)
    {}

    bool operator ==(const BatchNormalizationDescriptor& rhs) const
    {
        return m_Eps == rhs.m_Eps && m_DataLayout == rhs.m_DataLayout;
    }

    /// Value to add to the variance. Used to avoid dividing by zero.
    float m_Eps;
    /// The data layout to be used (NCHW, NHWC).
    DataLayout m_DataLayout;
};

/// An InstanceNormalizationDescriptor for InstanceNormalizationLayer
struct InstanceNormalizationDescriptor : BaseDescriptor
{
    InstanceNormalizationDescriptor()
        : m_Gamma(1.0f)
        , m_Beta(0.0f)
        , m_Eps(1e-12f)
        , m_DataLayout(DataLayout::NCHW)
    {}

    bool operator ==(const InstanceNormalizationDescriptor& rhs) const
    {
        return m_Gamma      == rhs.m_Gamma &&
               m_Beta       == rhs.m_Beta &&
               m_Eps        == rhs.m_Eps &&
               m_DataLayout == rhs.m_DataLayout;
    }

    /// Gamma, the scale scalar value applied for the normalized tensor. Defaults to 1.0.
    float m_Gamma;
    /// Beta, the offset scalar value applied for the normalized tensor. Defaults to 1.0.
    float m_Beta;
    /// Epsilon, small scalar value added to variance to avoid dividing by zero. Defaults to 1e-12f.
    float m_Eps;
    /// The data layout to be used (NCHW, NHWC).
    DataLayout m_DataLayout;
};

/// A BatchToSpaceNdDescriptor for the BatchToSpaceNdLayer.
struct BatchToSpaceNdDescriptor : BaseDescriptor
{
    BatchToSpaceNdDescriptor()
        : m_BlockShape({1, 1})
        , m_Crops({{0, 0}, {0, 0}})
        , m_DataLayout(DataLayout::NCHW)
    {}

    BatchToSpaceNdDescriptor(std::vector<unsigned int> blockShape,
                             std::vector<std::pair<unsigned int, unsigned int>> crops)
        : m_BlockShape(blockShape)
        , m_Crops(crops)
        , m_DataLayout(DataLayout::NCHW)
    {}

    bool operator ==(const BatchToSpaceNdDescriptor& rhs) const
    {
        return m_BlockShape == rhs.m_BlockShape &&
               m_Crops      == rhs.m_Crops &&
               m_DataLayout == rhs.m_DataLayout;
    }

    /// Block shape values.
    std::vector<unsigned int> m_BlockShape;
    /// The values to crop from the input dimension.
    std::vector<std::pair<unsigned int, unsigned int>> m_Crops;
    /// The data layout to be used (NCHW, NHWC).
    DataLayout m_DataLayout;
};

/// A FakeQuantizationDescriptor for the FakeQuantizationLayer.
struct FakeQuantizationDescriptor : BaseDescriptor
{
        FakeQuantizationDescriptor()
        : m_Min(-6.0f)
        , m_Max(6.0f)
    {}

    bool operator ==(const FakeQuantizationDescriptor& rhs) const
    {
        return m_Min == rhs.m_Min && m_Max == rhs.m_Max;
    }

    /// Minimum value.
    float m_Min;
    /// Maximum value.
    float m_Max;
};

/// A FillDescriptor for the FillLayer
struct FillDescriptor : BaseDescriptor
{
    FillDescriptor()
    : m_Value(0)
    {}

    FillDescriptor(const float& value)
    : m_Value(value)
    {}

    bool operator ==(const FillDescriptor& rhs) const
    {
        return m_Value == rhs.m_Value;
    }

    float m_Value;
};

/// A GatherDescriptor for the GatherLayer.
struct GatherDescriptor : BaseDescriptor
{
    GatherDescriptor()
        : m_Axis(0)
    {}

    GatherDescriptor(int32_t axis)
        : m_Axis(axis)
    {}

    bool operator ==(const GatherDescriptor& rhs) const
    {
        return m_Axis == rhs.m_Axis;
    }

    /// The axis in params to gather indices from
    int32_t m_Axis;
};

/// A ResizeBilinearDescriptor for the ResizeBilinearLayer.
struct ResizeBilinearDescriptor : BaseDescriptor
{
    ResizeBilinearDescriptor()
        : m_TargetWidth(0)
        , m_TargetHeight(0)
        , m_DataLayout(DataLayout::NCHW)
        , m_AlignCorners(false)
        , m_HalfPixelCenters(false)
    {}

    bool operator ==(const ResizeBilinearDescriptor& rhs) const
    {
        return m_TargetWidth          == rhs.m_TargetWidth &&
               m_TargetHeight         == rhs.m_TargetHeight &&
               m_DataLayout           == rhs.m_DataLayout &&
               m_AlignCorners         == rhs.m_AlignCorners &&
               m_HalfPixelCenters     == rhs.m_HalfPixelCenters;
    }

    /// Target width value.
    uint32_t          m_TargetWidth;
    /// Target height value.
    uint32_t          m_TargetHeight;
    /// The data layout to be used (NCHW, NHWC).
    DataLayout m_DataLayout;
    /// Aligned corners
    bool m_AlignCorners;
    /// Half Pixel Centers
    bool m_HalfPixelCenters;
};

/// A ResizeDescriptor for the ResizeLayer.
struct ResizeDescriptor : BaseDescriptor
{
    ResizeDescriptor()
        : m_TargetWidth(0)
        , m_TargetHeight(0)
        , m_Method(ResizeMethod::NearestNeighbor)
        , m_DataLayout(DataLayout::NCHW)
        , m_AlignCorners(false)
        , m_HalfPixelCenters(false)
    {}

    bool operator ==(const ResizeDescriptor& rhs) const
    {
        return m_TargetWidth          == rhs.m_TargetWidth &&
               m_TargetHeight         == rhs.m_TargetHeight &&
               m_Method               == rhs.m_Method &&
               m_DataLayout           == rhs.m_DataLayout &&
               m_AlignCorners         == rhs.m_AlignCorners &&
               m_HalfPixelCenters     == rhs.m_HalfPixelCenters;
    }

    /// Target width value.
    uint32_t m_TargetWidth;
    /// Target height value.
    uint32_t m_TargetHeight;
    /// The Interpolation method to use
    /// (Bilinear, NearestNeighbor).
    ResizeMethod m_Method;
    /// The data layout to be used (NCHW, NHWC).
    DataLayout m_DataLayout;
    /// Aligned corners
    bool m_AlignCorners;
    /// Half Pixel Centers
    bool m_HalfPixelCenters;
};


/// A ReshapeDescriptor for the ReshapeLayer.
struct ReshapeDescriptor : BaseDescriptor
{
    ReshapeDescriptor()
        : m_TargetShape()
    {}

    ReshapeDescriptor(const TensorShape& shape)
        : m_TargetShape(shape)
    {}

    bool operator ==(const ReshapeDescriptor& rhs) const
    {
        return m_TargetShape == rhs.m_TargetShape;
    }

    /// Target shape value.
    TensorShape m_TargetShape;
};

/// A SpaceToBatchNdDescriptor for the SpaceToBatchNdLayer.
struct SpaceToBatchNdDescriptor : BaseDescriptor
{
    SpaceToBatchNdDescriptor()
        : m_BlockShape({1, 1})
        , m_PadList({{0, 0}, {0, 0}})
        , m_DataLayout(DataLayout::NCHW)
    {}

    SpaceToBatchNdDescriptor(const std::vector<unsigned int>& blockShape,
                             const std::vector<std::pair<unsigned int, unsigned int>>& padList)
        : m_BlockShape(blockShape)
        , m_PadList(padList)
        , m_DataLayout(DataLayout::NCHW)
    {}

    bool operator ==(const SpaceToBatchNdDescriptor& rhs) const
    {
        return m_BlockShape == rhs.m_BlockShape &&
               m_PadList    == rhs.m_PadList &&
               m_DataLayout == rhs.m_DataLayout;
    }

    /// Block shape value.
    std::vector<unsigned int> m_BlockShape;
    /// @brief Specifies the padding values for the input dimension:
    /// heightPad{top, bottom} widthPad{left, right}.
    std::vector<std::pair<unsigned int, unsigned int>> m_PadList;
    /// The data layout to be used (NCHW, NHWC).
    DataLayout m_DataLayout;
};

/// A SpaceToDepthDescriptor for the SpaceToDepthLayer
struct SpaceToDepthDescriptor : BaseDescriptor
{
    SpaceToDepthDescriptor()
        : SpaceToDepthDescriptor(1u, DataLayout::NHWC)
    {}

    SpaceToDepthDescriptor(unsigned int blockSize, DataLayout dataLayout)
        : m_BlockSize(blockSize)
        , m_DataLayout(dataLayout)
    {}

    bool operator ==(const SpaceToDepthDescriptor& rhs) const
    {
        return m_BlockSize == rhs.m_BlockSize && m_DataLayout == rhs.m_DataLayout;
    }

    /// Scalar specifying the input block size. It must be >= 1
    unsigned int m_BlockSize;

    /// The data layout to be used (NCHW, NHWC).
    DataLayout m_DataLayout;
};

/// A DepthToSpaceDescriptor for the DepthToSpaceLayer
using DepthToSpaceDescriptor = SpaceToDepthDescriptor;

/// An LstmDescriptor for the LstmLayer.
struct LstmDescriptor : BaseDescriptor
{
    LstmDescriptor()
        : m_ActivationFunc(1) // 0: None, 1: Relu, 3: Relu6, 4: Tanh, 6: Sigmoid
        , m_ClippingThresCell(0.0)
        , m_ClippingThresProj(0.0)
        , m_CifgEnabled(true)
        , m_PeepholeEnabled(false)
        , m_ProjectionEnabled(false)
        , m_LayerNormEnabled(false)
    {}

    bool operator ==(const LstmDescriptor& rhs) const
    {
        return m_ActivationFunc    == rhs.m_ActivationFunc &&
               m_ClippingThresCell == rhs.m_ClippingThresCell &&
               m_ClippingThresProj == rhs.m_ClippingThresProj &&
               m_CifgEnabled       == rhs.m_CifgEnabled &&
               m_PeepholeEnabled   == rhs.m_PeepholeEnabled &&
               m_LayerNormEnabled  == rhs.m_LayerNormEnabled;
    }

    /// @brief The activation function to use.
    /// 0: None, 1: Relu, 3: Relu6, 4: Tanh, 6: Sigmoid.
    uint32_t m_ActivationFunc;
    /// Clipping threshold value for the cell state.
    float m_ClippingThresCell;
    /// Clipping threshold value for the projection.
    float m_ClippingThresProj;
    /// Enable/disable cifg (coupled input & forget gate).
    bool m_CifgEnabled;
    /// Enable/disable peephole.
    bool m_PeepholeEnabled;
    /// Enable/disable the projection layer.
    bool m_ProjectionEnabled;
    /// Enable/disable layer normalization
    bool m_LayerNormEnabled;
};

/// A MeanDescriptor for the MeanLayer.
struct MeanDescriptor : BaseDescriptor
{
    MeanDescriptor()
        : m_Axis()
        , m_KeepDims(false)
    {}

    MeanDescriptor(const std::vector<unsigned int>& axis, bool keepDims)
        : m_Axis(axis)
        , m_KeepDims(keepDims)
    {}

    bool operator ==(const MeanDescriptor& rhs) const
    {
        return m_Axis == rhs.m_Axis && m_KeepDims == rhs.m_KeepDims;
    }

    /// Values for the dimensions to reduce.
    std::vector<unsigned int> m_Axis;
    /// Enable/disable keep dimensions. If true, then the reduced dimensions that are of length 1 are kept.
    bool m_KeepDims;
};

/// A PadDescriptor for the PadLayer.
struct PadDescriptor : BaseDescriptor
{
    PadDescriptor() : m_PadValue(0)
    {}

    PadDescriptor(const std::vector<std::pair<unsigned int, unsigned int>>& padList, const float& padValue = 0)
        : m_PadList(padList)
        , m_PadValue(padValue)
    {}

    bool operator ==(const PadDescriptor& rhs) const
    {
        return m_PadList == rhs.m_PadList && m_PadValue == rhs.m_PadValue;
    }

    /// @brief Specifies the padding for input dimension.
    /// First is the number of values to add before the tensor in the dimension.
    /// Second is the number of values to add after the tensor in the dimension.
    /// The number of pairs should match the number of dimensions in the input tensor.
    std::vector<std::pair<unsigned int, unsigned int>> m_PadList;

    /// Optional value to use for padding, defaults to 0
    float m_PadValue;
};

/// A SliceDescriptor for the SliceLayer.
struct SliceDescriptor : BaseDescriptor
{
    SliceDescriptor(const std::vector<unsigned int>& begin, const std::vector<unsigned int>& size)
        : m_Begin(begin)
        , m_Size(size)
    {}

    SliceDescriptor() : SliceDescriptor({}, {})
    {}

    bool operator ==(const SliceDescriptor& rhs) const
    {
        return m_Begin == rhs.m_Begin && m_Size == rhs.m_Size;
    }

    /// Beginning indices of the slice in each dimension.
    std::vector<unsigned int> m_Begin;

    /// Size of the slice in each dimension.
    std::vector<unsigned int> m_Size;
};

/// A StackDescriptor for the StackLayer.
struct StackDescriptor : BaseDescriptor
{
    StackDescriptor()
        : m_Axis(0)
        , m_NumInputs(0)
        , m_InputShape()
    {}

    StackDescriptor(uint32_t axis, uint32_t numInputs, const TensorShape& inputShape)
        : m_Axis(axis)
        , m_NumInputs(numInputs)
        , m_InputShape(inputShape)
    {}

    bool operator ==(const StackDescriptor& rhs) const
    {
        return m_Axis       == rhs.m_Axis &&
               m_NumInputs  == rhs.m_NumInputs &&
               m_InputShape == rhs.m_InputShape;
    }

    /// 0-based axis along which to stack the input tensors.
    uint32_t m_Axis;
    /// Number of input tensors.
    uint32_t m_NumInputs;
    /// Required shape of all input tensors.
    TensorShape m_InputShape;
};

/// A StandInDescriptor for the StandIn layer
struct StandInDescriptor : BaseDescriptor
{
    StandInDescriptor() {};

    StandInDescriptor(uint32_t numInputs, uint32_t numOutputs)
        : m_NumInputs(numInputs)
        , m_NumOutputs(numOutputs)
    {}

    bool operator ==(const StandInDescriptor& rhs) const
    {
        return m_NumInputs  == rhs.m_NumInputs &&
               m_NumOutputs == rhs.m_NumOutputs;
    }

    /// Number of input tensors
    uint32_t m_NumInputs = 0;
    /// Number of output tensors
    uint32_t m_NumOutputs = 0;
};

/// A StridedSliceDescriptor for the StridedSliceLayer.
struct StridedSliceDescriptor : BaseDescriptor
{
    StridedSliceDescriptor(const std::vector<int>& begin,
                           const std::vector<int>& end,
                           const std::vector<int>& stride)
        : m_Begin(begin)
        , m_End(end)
        , m_Stride(stride)
        , m_BeginMask(0)
        , m_EndMask(0)
        , m_ShrinkAxisMask(0)
        , m_EllipsisMask(0)
        , m_NewAxisMask(0)
        , m_DataLayout(DataLayout::NCHW)
    {}

    StridedSliceDescriptor()
        : StridedSliceDescriptor({}, {}, {})
    {}

    bool operator ==(const StridedSliceDescriptor& rhs) const
    {
        return m_Begin          == rhs.m_Begin &&
               m_End            == rhs.m_End &&
               m_Stride         == rhs.m_Stride &&
               m_BeginMask      == rhs.m_BeginMask &&
               m_EndMask        == rhs.m_EndMask &&
               m_ShrinkAxisMask == rhs.m_ShrinkAxisMask &&
               m_EllipsisMask   == rhs.m_EllipsisMask &&
               m_NewAxisMask    == rhs.m_NewAxisMask &&
               m_DataLayout     == rhs.m_DataLayout;
    }

    int GetStartForAxis(const TensorShape& inputShape, unsigned int axis) const;
    int GetStopForAxis(const TensorShape& inputShape,
                       unsigned int axis,
                       int startForAxis) const;

    /// Begin values for the input that will be sliced.
    std::vector<int> m_Begin;
    /// End values for the input that will be sliced.
    std::vector<int> m_End;
    /// Stride values for the input that will be sliced.
    std::vector<int> m_Stride;

    /// @brief Begin mask value. If set, then the begin is disregarded and the fullest
    /// range is used for the dimension.
    int32_t m_BeginMask;
    /// @brief End mask value. If set, then the end is disregarded and the fullest range
    /// is used for the dimension.
    int32_t m_EndMask;
    /// Shrink axis mask value. If set, the nth specification shrinks the dimensionality by 1.
    int32_t m_ShrinkAxisMask;
    /// Ellipsis mask value.
    int32_t m_EllipsisMask;
    /// @brief New axis mask value. If set, the begin, end and stride is disregarded and
    /// a new 1 dimension is inserted to this location of the output tensor.
    int32_t m_NewAxisMask;

    /// The data layout to be used (NCHW, NHWC).
    DataLayout m_DataLayout;
};

/// A PreCompiledDescriptor for the PreCompiledLayer.
struct PreCompiledDescriptor : BaseDescriptor
{
    PreCompiledDescriptor(unsigned int numInputSlots = 1u, unsigned int numOutputSlots = 1u)
        : m_NumInputSlots(numInputSlots), m_NumOutputSlots(numOutputSlots)
    {}

    ~PreCompiledDescriptor() = default;

    unsigned int m_NumInputSlots;
    unsigned int m_NumOutputSlots;
};

/// A QLstmDescriptor for the QLstmLayer.
struct QLstmDescriptor : BaseDescriptor
{
    QLstmDescriptor()
            : m_CellClip(0.0)
            , m_ProjectionClip(0.0)
            , m_CifgEnabled(true)
            , m_PeepholeEnabled(false)
            , m_ProjectionEnabled(false)
            , m_LayerNormEnabled(false)
            , m_InputIntermediateScale(0.0)
            , m_ForgetIntermediateScale(0.0)
            , m_CellIntermediateScale(0.0)
            , m_OutputIntermediateScale(0.0)
            , m_HiddenStateZeroPoint(0)
            , m_HiddenStateScale(0.0)
    {}

    bool operator ==(const QLstmDescriptor& rhs) const
    {
        return m_CellClip          == rhs.m_CellClip &&
               m_ProjectionClip    == rhs.m_ProjectionClip &&
               m_CifgEnabled       == rhs.m_CifgEnabled &&
               m_PeepholeEnabled   == rhs.m_PeepholeEnabled &&
               m_ProjectionEnabled == rhs.m_ProjectionEnabled &&
               m_LayerNormEnabled  == rhs.m_LayerNormEnabled &&
               m_InputIntermediateScale == rhs.m_InputIntermediateScale &&
               m_ForgetIntermediateScale == rhs.m_ForgetIntermediateScale &&
               m_CellIntermediateScale == rhs.m_CellIntermediateScale &&
               m_OutputIntermediateScale == rhs.m_OutputIntermediateScale &&
               m_HiddenStateZeroPoint == rhs.m_HiddenStateZeroPoint &&
               m_HiddenStateScale == rhs.m_HiddenStateScale;
    }

    /// Clipping threshold value for the cell state
    float m_CellClip;
    /// Clipping threshold value for the projection
    float m_ProjectionClip;
    /// Enable/disable CIFG (coupled input & forget gate).
    bool m_CifgEnabled;
    /// Enable/disable peephole
    bool m_PeepholeEnabled;
    /// Enable/disable the projection layer
    bool m_ProjectionEnabled;
    /// Enable/disable layer normalization
    bool m_LayerNormEnabled;
    /// Input intermediate quantization scale
    float m_InputIntermediateScale;
    /// Forget intermediate quantization scale
    float m_ForgetIntermediateScale;
    /// Cell intermediate quantization scale
    float m_CellIntermediateScale;
    /// Output intermediate quantization scale
    float m_OutputIntermediateScale;
    /// Hidden State zero point
    int32_t m_HiddenStateZeroPoint;
    /// Hidden State quantization scale
    float m_HiddenStateScale;
};

/// A TransposeConvolution2dDescriptor for the TransposeConvolution2dLayer.
struct TransposeConvolution2dDescriptor : BaseDescriptor
{
    TransposeConvolution2dDescriptor() :
        m_PadLeft(0),
        m_PadRight(0),
        m_PadTop(0),
        m_PadBottom(0),
        m_StrideX(0),
        m_StrideY(0),
        m_BiasEnabled(false),
        m_DataLayout(DataLayout::NCHW),
        m_OutputShapeEnabled(false)
    {}

    bool operator ==(const TransposeConvolution2dDescriptor& rhs) const
    {
        return m_PadLeft            == rhs.m_PadLeft &&
               m_PadRight           == rhs.m_PadRight &&
               m_PadTop             == rhs.m_PadTop &&
               m_PadBottom          == rhs.m_PadBottom &&
               m_StrideX            == rhs.m_StrideX &&
               m_StrideY            == rhs.m_StrideY &&
               m_BiasEnabled        == rhs.m_BiasEnabled &&
               m_DataLayout         == rhs.m_DataLayout &&
               m_OutputShapeEnabled == rhs.m_OutputShapeEnabled &&
               m_OutputShape        == rhs.m_OutputShape;
    }

    /// Padding left value in the width dimension.
    uint32_t                  m_PadLeft;
    /// Padding right value in the width dimension.
    uint32_t                  m_PadRight;
    /// Padding top value in the height dimension.
    uint32_t                  m_PadTop;
    /// Padding bottom value in the height dimension.
    uint32_t                  m_PadBottom;
    /// Stride value when proceeding through input for the width dimension.
    uint32_t                  m_StrideX;
    /// Stride value when proceeding through input for the height dimension.
    uint32_t                  m_StrideY;
    /// Enable/disable bias.
    bool                      m_BiasEnabled;
    /// The data layout to be used (NCHW, NHWC).
    DataLayout                m_DataLayout;
    /// Output shape if it has been specified.
    bool                      m_OutputShapeEnabled;
    std::vector<unsigned int> m_OutputShape;
};

/// A TransposeDescriptor for the TransposeLayer.
struct TransposeDescriptor : BaseDescriptor
{
    TransposeDescriptor()
            : m_DimMappings{}
    {}

    TransposeDescriptor(const PermutationVector& dimMappings)
            : m_DimMappings(dimMappings)
    {}

    bool operator ==(const TransposeDescriptor &rhs) const
    {
        return m_DimMappings.IsEqual(rhs.m_DimMappings);
    }

    /// @brief Indicates how to translate tensor elements from a given source into the target destination, when
    /// source and target potentially have different memory layouts e.g. {0U, 3U, 1U, 2U}.
    PermutationVector m_DimMappings;
};

/// A LogicalBinaryDescriptor for the LogicalBinaryLayer
struct LogicalBinaryDescriptor : BaseDescriptor
{
    LogicalBinaryDescriptor()
        : LogicalBinaryDescriptor(LogicalBinaryOperation::LogicalAnd)
    {}

    LogicalBinaryDescriptor(LogicalBinaryOperation operation)
        : m_Operation(operation)
    {}

    bool operator ==(const LogicalBinaryDescriptor &rhs) const
    {
        return m_Operation == rhs.m_Operation;
    }

    /// Specifies the logical operation to execute
    LogicalBinaryOperation m_Operation;
};

/// A ReduceDescriptor for the REDUCE operators.
struct ReduceDescriptor : BaseDescriptor
{
    ReduceDescriptor()
        : m_KeepDims(false)
        , m_vAxis()
        , m_ReduceOperation(ReduceOperation::Sum)
    {}

    bool operator ==(const ReduceDescriptor& rhs) const
    {
        return m_KeepDims             == rhs.m_KeepDims &&
               m_vAxis                == rhs.m_vAxis &&
               m_ReduceOperation      == rhs.m_ReduceOperation;
    }

    /// if true then output shape has no change.
    bool m_KeepDims;
    /// The indices of the dimensions to reduce.
    std::vector<uint32_t> m_vAxis;
    /// Specifies the reduction operation to execute
    ReduceOperation m_ReduceOperation;
};

} // namespace armnn
