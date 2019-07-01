//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/ArmNN.hpp>
#include "TestLayerVisitor.hpp"
#include <boost/test/unit_test.hpp>

namespace armnn
{

// Concrete TestLayerVisitor subclasses for layers taking Descriptor argument with overridden VisitLayer methods
class TestPermuteLayerVisitor : public TestLayerVisitor
{
private:
    const PermuteDescriptor m_VisitorDescriptor;

public:
    explicit TestPermuteLayerVisitor(const PermuteDescriptor& permuteDescriptor, const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_VisitorDescriptor(permuteDescriptor.m_DimMappings)
    {};

    void CheckDescriptor(const PermuteDescriptor& permuteDescriptor)
    {
        if (permuteDescriptor.m_DimMappings.GetSize() == m_VisitorDescriptor.m_DimMappings.GetSize())
        {
            for (unsigned int i = 0; i < permuteDescriptor.m_DimMappings.GetSize(); ++i)
            {
                BOOST_CHECK_EQUAL(permuteDescriptor.m_DimMappings[i], m_VisitorDescriptor.m_DimMappings[i]);
            }
        }
        else
        {
            BOOST_ERROR("Unequal vector size for batchToSpaceNdDescriptor m_DimMappings.");
        }
    };

    void VisitPermuteLayer(const IConnectableLayer* layer,
                           const PermuteDescriptor& permuteDescriptor,
                           const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckDescriptor(permuteDescriptor);
        CheckLayerName(name);
    };
};

class TestBatchToSpaceNdLayerVisitor : public TestLayerVisitor
{
private:
    BatchToSpaceNdDescriptor m_VisitorDescriptor;

public:
    explicit TestBatchToSpaceNdLayerVisitor(const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                            const char* name = nullptr)
            : TestLayerVisitor(name)
            , m_VisitorDescriptor(batchToSpaceNdDescriptor.m_BlockShape, batchToSpaceNdDescriptor.m_Crops)
    {
        m_VisitorDescriptor.m_DataLayout = batchToSpaceNdDescriptor.m_DataLayout;
    };

    void CheckDescriptor(const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor)
    {
        if (batchToSpaceNdDescriptor.m_BlockShape.size() == m_VisitorDescriptor.m_BlockShape.size())
        {
            for (unsigned int i = 0; i < batchToSpaceNdDescriptor.m_BlockShape.size(); ++i)
            {
                BOOST_CHECK_EQUAL(batchToSpaceNdDescriptor.m_BlockShape[i], m_VisitorDescriptor.m_BlockShape[i]);
            }
        }
        else
        {
            BOOST_ERROR("Unequal vector size for batchToSpaceNdDescriptor m_BlockShape.");
        }

        if (batchToSpaceNdDescriptor.m_Crops.size() == m_VisitorDescriptor.m_Crops.size())
        {
            for (unsigned int i = 0; i < batchToSpaceNdDescriptor.m_Crops.size(); ++i)
            {
                BOOST_CHECK_EQUAL(batchToSpaceNdDescriptor.m_Crops[i].first, m_VisitorDescriptor.m_Crops[i].first);
                BOOST_CHECK_EQUAL(batchToSpaceNdDescriptor.m_Crops[i].second, m_VisitorDescriptor.m_Crops[i].second);
            }
        }
        else
        {
            BOOST_ERROR("Unequal vector size for batchToSpaceNdDescriptor m_Crops.");
        }

        BOOST_CHECK(batchToSpaceNdDescriptor.m_DataLayout == m_VisitorDescriptor.m_DataLayout);
    }

    void VisitBatchToSpaceNdLayer(const IConnectableLayer* layer,
                                  const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                  const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckDescriptor(batchToSpaceNdDescriptor);
        CheckLayerName(name);
    };
};

class TestPooling2dLayerVisitor : public TestLayerVisitor
{
private:
    Pooling2dDescriptor m_VisitorDescriptor;

public:
    explicit TestPooling2dLayerVisitor(const Pooling2dDescriptor& pooling2dDescriptor, const char* name = nullptr)
        : TestLayerVisitor(name)
    {
        m_VisitorDescriptor.m_PoolType            = pooling2dDescriptor.m_PoolType;
        m_VisitorDescriptor.m_PadLeft             = pooling2dDescriptor.m_PadLeft;
        m_VisitorDescriptor.m_PadRight            = pooling2dDescriptor.m_PadRight;
        m_VisitorDescriptor.m_PadTop              = pooling2dDescriptor.m_PadTop;
        m_VisitorDescriptor.m_PadBottom           = pooling2dDescriptor.m_PadBottom;
        m_VisitorDescriptor.m_PoolWidth           = pooling2dDescriptor.m_PoolWidth;
        m_VisitorDescriptor.m_PoolHeight          = pooling2dDescriptor.m_PoolHeight;
        m_VisitorDescriptor.m_StrideX             = pooling2dDescriptor.m_StrideX;
        m_VisitorDescriptor.m_StrideY             = pooling2dDescriptor.m_StrideY;
        m_VisitorDescriptor.m_OutputShapeRounding = pooling2dDescriptor.m_OutputShapeRounding;
        m_VisitorDescriptor.m_PaddingMethod       = pooling2dDescriptor.m_PaddingMethod;
        m_VisitorDescriptor.m_DataLayout          = pooling2dDescriptor.m_DataLayout;
    };

    void CheckDescriptor(const Pooling2dDescriptor& pooling2dDescriptor)
    {
        BOOST_CHECK(pooling2dDescriptor.m_PoolType == m_VisitorDescriptor.m_PoolType);
        BOOST_CHECK_EQUAL(pooling2dDescriptor.m_PadLeft, m_VisitorDescriptor.m_PadLeft);
        BOOST_CHECK_EQUAL(pooling2dDescriptor.m_PadRight, m_VisitorDescriptor.m_PadRight);
        BOOST_CHECK_EQUAL(pooling2dDescriptor.m_PadTop, m_VisitorDescriptor.m_PadTop);
        BOOST_CHECK_EQUAL(pooling2dDescriptor.m_PadBottom, m_VisitorDescriptor.m_PadBottom);
        BOOST_CHECK_EQUAL(pooling2dDescriptor.m_PoolWidth, m_VisitorDescriptor.m_PoolWidth);
        BOOST_CHECK_EQUAL(pooling2dDescriptor.m_PoolHeight, m_VisitorDescriptor.m_PoolHeight);
        BOOST_CHECK_EQUAL(pooling2dDescriptor.m_StrideX, m_VisitorDescriptor.m_StrideX);
        BOOST_CHECK_EQUAL(pooling2dDescriptor.m_StrideY, m_VisitorDescriptor.m_StrideY);
        BOOST_CHECK(pooling2dDescriptor.m_OutputShapeRounding == m_VisitorDescriptor.m_OutputShapeRounding);
        BOOST_CHECK(pooling2dDescriptor.m_PaddingMethod == m_VisitorDescriptor.m_PaddingMethod);
        BOOST_CHECK(pooling2dDescriptor.m_DataLayout == m_VisitorDescriptor.m_DataLayout);
    }

    void VisitPooling2dLayer(const IConnectableLayer* layer,
                             const Pooling2dDescriptor& pooling2dDescriptor,
                             const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckDescriptor(pooling2dDescriptor);
        CheckLayerName(name);
    };
};

class TestActivationLayerVisitor : public TestLayerVisitor
{
private:
    ActivationDescriptor m_VisitorDescriptor;

public:
    explicit TestActivationLayerVisitor(const ActivationDescriptor& activationDescriptor, const char* name = nullptr)
        : TestLayerVisitor(name)
    {
        m_VisitorDescriptor.m_Function = activationDescriptor.m_Function;
        m_VisitorDescriptor.m_A = activationDescriptor.m_A;
        m_VisitorDescriptor.m_B = activationDescriptor.m_B;
    };

    void CheckDescriptor(const ActivationDescriptor& activationDescriptor)
    {
        BOOST_CHECK(activationDescriptor.m_Function == m_VisitorDescriptor.m_Function);
        BOOST_CHECK_EQUAL(activationDescriptor.m_A, m_VisitorDescriptor.m_A);
        BOOST_CHECK_EQUAL(activationDescriptor.m_B, m_VisitorDescriptor.m_B);
    };

    void VisitActivationLayer(const IConnectableLayer* layer,
                              const ActivationDescriptor& activationDescriptor,
                              const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckDescriptor(activationDescriptor);
        CheckLayerName(name);
    };
};

class TestNormalizationLayerVisitor : public TestLayerVisitor
{
private:
    NormalizationDescriptor m_VisitorDescriptor;

public:
    explicit TestNormalizationLayerVisitor(const NormalizationDescriptor& normalizationDescriptor,
                                           const char* name = nullptr)
        : TestLayerVisitor(name)
    {
        m_VisitorDescriptor.m_NormChannelType = normalizationDescriptor.m_NormChannelType;
        m_VisitorDescriptor.m_NormMethodType  = normalizationDescriptor.m_NormMethodType;
        m_VisitorDescriptor.m_NormSize        = normalizationDescriptor.m_NormSize;
        m_VisitorDescriptor.m_Alpha           = normalizationDescriptor.m_Alpha;
        m_VisitorDescriptor.m_Beta            = normalizationDescriptor.m_Beta;
        m_VisitorDescriptor.m_K               = normalizationDescriptor.m_K;
        m_VisitorDescriptor.m_DataLayout      = normalizationDescriptor.m_DataLayout;
    };

    void CheckDescriptor(const NormalizationDescriptor& normalizationDescriptor)
    {
        BOOST_CHECK(normalizationDescriptor.m_NormChannelType == m_VisitorDescriptor.m_NormChannelType);
        BOOST_CHECK(normalizationDescriptor.m_NormMethodType == m_VisitorDescriptor.m_NormMethodType);
        BOOST_CHECK_EQUAL(normalizationDescriptor.m_NormSize, m_VisitorDescriptor.m_NormSize);
        BOOST_CHECK_EQUAL(normalizationDescriptor.m_Alpha, m_VisitorDescriptor.m_Alpha);
        BOOST_CHECK_EQUAL(normalizationDescriptor.m_Beta, m_VisitorDescriptor.m_Beta);
        BOOST_CHECK_EQUAL(normalizationDescriptor.m_K, m_VisitorDescriptor.m_K);
        BOOST_CHECK(normalizationDescriptor.m_DataLayout == m_VisitorDescriptor.m_DataLayout);
    }

    void VisitNormalizationLayer(const IConnectableLayer* layer,
                                 const NormalizationDescriptor& normalizationDescriptor,
                                 const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckDescriptor(normalizationDescriptor);
        CheckLayerName(name);
    };
};

class TestSoftmaxLayerVisitor : public TestLayerVisitor
{
private:
    SoftmaxDescriptor m_VisitorDescriptor;

public:
    explicit TestSoftmaxLayerVisitor(const SoftmaxDescriptor& softmaxDescriptor, const char* name = nullptr)
        : TestLayerVisitor(name)
    {
        m_VisitorDescriptor.m_Beta = softmaxDescriptor.m_Beta;
    };

    void CheckDescriptor(const SoftmaxDescriptor& softmaxDescriptor)
    {
        BOOST_CHECK_EQUAL(softmaxDescriptor.m_Beta, m_VisitorDescriptor.m_Beta);
    }

    void VisitSoftmaxLayer(const IConnectableLayer* layer,
                           const SoftmaxDescriptor& softmaxDescriptor,
                           const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckDescriptor(softmaxDescriptor);
        CheckLayerName(name);
    };
};

class TestSplitterLayerVisitor : public TestLayerVisitor
{
private:
    ViewsDescriptor m_VisitorDescriptor;

public:
    explicit TestSplitterLayerVisitor(const ViewsDescriptor& splitterDescriptor, const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_VisitorDescriptor(splitterDescriptor.GetNumViews(), splitterDescriptor.GetNumDimensions())
    {
        if (splitterDescriptor.GetNumViews() != m_VisitorDescriptor.GetNumViews())
        {
            BOOST_ERROR("Unequal number of views in splitter descriptor.");
        }
        else if (splitterDescriptor.GetNumDimensions() != m_VisitorDescriptor.GetNumDimensions())
        {
            BOOST_ERROR("Unequal number of dimensions in splitter descriptor.");
        }
        else
        {
            for (unsigned int i = 0; i < splitterDescriptor.GetNumViews(); ++i)
            {
                for (unsigned int j = 0; j < splitterDescriptor.GetNumDimensions(); ++j)
                {
                    m_VisitorDescriptor.SetViewOriginCoord(i, j, splitterDescriptor.GetViewOrigin(i)[j]);
                    m_VisitorDescriptor.SetViewSize(i, j, splitterDescriptor.GetViewSizes(i)[j]);
                }
            }
        }
    };

    void CheckDescriptor(const ViewsDescriptor& splitterDescriptor)
    {

        BOOST_CHECK_EQUAL(splitterDescriptor.GetNumViews(), m_VisitorDescriptor.GetNumViews());
        BOOST_CHECK_EQUAL(splitterDescriptor.GetNumDimensions(), m_VisitorDescriptor.GetNumDimensions());

        if (splitterDescriptor.GetNumViews() != m_VisitorDescriptor.GetNumViews())
        {
            BOOST_ERROR("Unequal number of views in splitter descriptor.");
        }
        else if (splitterDescriptor.GetNumDimensions() != m_VisitorDescriptor.GetNumDimensions())
        {
            BOOST_ERROR("Unequal number of dimensions in splitter descriptor.");
        }
        else
        {
            for (unsigned int i = 0; i < splitterDescriptor.GetNumViews(); ++i)
            {
                for (unsigned int j = 0; j < splitterDescriptor.GetNumDimensions(); ++j)
                {
                    BOOST_CHECK_EQUAL(splitterDescriptor.GetViewOrigin(i)[j], m_VisitorDescriptor.GetViewOrigin(i)[j]);
                    BOOST_CHECK_EQUAL(splitterDescriptor.GetViewSizes(i)[j], m_VisitorDescriptor.GetViewSizes(i)[j]);
                }
            }
        }
    };

    void VisitSplitterLayer(const IConnectableLayer* layer,
                            const ViewsDescriptor& splitterDescriptor,
                            const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckDescriptor(splitterDescriptor);
        CheckLayerName(name);
    };
};

class TestConcatLayerVisitor : public TestLayerVisitor
{
private:
    OriginsDescriptor m_VisitorDescriptor;

public:
    explicit TestConcatLayerVisitor(const OriginsDescriptor& concatDescriptor, const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_VisitorDescriptor(concatDescriptor.GetNumViews(), concatDescriptor.GetNumDimensions())
    {
        m_VisitorDescriptor.SetConcatAxis(concatDescriptor.GetConcatAxis());

        if (concatDescriptor.GetNumViews() != m_VisitorDescriptor.GetNumViews())
        {
            BOOST_ERROR("Unequal number of views in splitter descriptor.");
        }
        else if (concatDescriptor.GetNumDimensions() != m_VisitorDescriptor.GetNumDimensions())
        {
            BOOST_ERROR("Unequal number of dimensions in splitter descriptor.");
        }
        else
        {
            for (unsigned int i = 0; i < concatDescriptor.GetNumViews(); ++i)
            {
                for (unsigned int j = 0; j < concatDescriptor.GetNumDimensions(); ++j)
                {
                    m_VisitorDescriptor.SetViewOriginCoord(i, j, concatDescriptor.GetViewOrigin(i)[j]);
                }
            }
        }
    };

    void CheckDescriptor(const OriginsDescriptor& concatDescriptor)
    {
        BOOST_CHECK_EQUAL(concatDescriptor.GetNumViews(), m_VisitorDescriptor.GetNumViews());
        BOOST_CHECK_EQUAL(concatDescriptor.GetNumDimensions(), m_VisitorDescriptor.GetNumDimensions());
        BOOST_CHECK_EQUAL(concatDescriptor.GetConcatAxis(), m_VisitorDescriptor.GetConcatAxis());

        if (concatDescriptor.GetNumViews() != m_VisitorDescriptor.GetNumViews())
        {
            BOOST_ERROR("Unequal number of views in splitter descriptor.");
        }
        else if (concatDescriptor.GetNumDimensions() != m_VisitorDescriptor.GetNumDimensions())
        {
            BOOST_ERROR("Unequal number of dimensions in splitter descriptor.");
        }
        else
        {
            for (unsigned int i = 0; i < concatDescriptor.GetNumViews(); ++i)
            {
                for (unsigned int j = 0; j < concatDescriptor.GetNumDimensions(); ++j)
                {
                    BOOST_CHECK_EQUAL(concatDescriptor.GetViewOrigin(i)[j], m_VisitorDescriptor.GetViewOrigin(i)[j]);
                }
            }
        }
    }

    void VisitConcatLayer(const IConnectableLayer* layer,
                          const OriginsDescriptor& concatDescriptor,
                          const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckDescriptor(concatDescriptor);
        CheckLayerName(name);
    };
};

class TestResizeLayerVisitor : public TestLayerVisitor
{
private:
    ResizeDescriptor m_VisitorDescriptor;

public:
    explicit TestResizeLayerVisitor(const ResizeDescriptor& descriptor, const char* name = nullptr)
        : TestLayerVisitor(name)
    {
        m_VisitorDescriptor.m_Method       = descriptor.m_Method;
        m_VisitorDescriptor.m_TargetWidth  = descriptor.m_TargetWidth;
        m_VisitorDescriptor.m_TargetHeight = descriptor.m_TargetHeight;
        m_VisitorDescriptor.m_DataLayout   = descriptor.m_DataLayout;
    };

    void CheckDescriptor(const ResizeDescriptor& descriptor)
    {
        BOOST_CHECK(descriptor.m_Method       == m_VisitorDescriptor.m_Method);
        BOOST_CHECK(descriptor.m_TargetWidth  == m_VisitorDescriptor.m_TargetWidth);
        BOOST_CHECK(descriptor.m_TargetHeight == m_VisitorDescriptor.m_TargetHeight);
        BOOST_CHECK(descriptor.m_DataLayout   == m_VisitorDescriptor.m_DataLayout);
    }

    void VisitResizeLayer(const IConnectableLayer* layer,
                          const ResizeDescriptor& descriptor,
                          const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckDescriptor(descriptor);
        CheckLayerName(name);
    };
};

class TestL2NormalizationLayerVisitor : public TestLayerVisitor
{
private:
    L2NormalizationDescriptor m_VisitorDescriptor;

public:
    explicit TestL2NormalizationLayerVisitor(const L2NormalizationDescriptor& desc, const char* name = nullptr)
        : TestLayerVisitor(name)
    {
        m_VisitorDescriptor.m_DataLayout  = desc.m_DataLayout;
    };

    void CheckDescriptor(const L2NormalizationDescriptor& desc)
    {
        BOOST_CHECK(desc.m_DataLayout == m_VisitorDescriptor.m_DataLayout);
    }

    void VisitL2NormalizationLayer(const IConnectableLayer* layer,
                                   const L2NormalizationDescriptor& desc,
                                   const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckDescriptor(desc);
        CheckLayerName(name);
    };
};

class TestReshapeLayerVisitor : public TestLayerVisitor
{
private:
    const ReshapeDescriptor m_VisitorDescriptor;

public:
    explicit TestReshapeLayerVisitor(const ReshapeDescriptor& reshapeDescriptor, const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_VisitorDescriptor(reshapeDescriptor.m_TargetShape)
    {};

    void CheckDescriptor(const ReshapeDescriptor& reshapeDescriptor)
    {
        BOOST_CHECK_MESSAGE(reshapeDescriptor.m_TargetShape == m_VisitorDescriptor.m_TargetShape,
                            reshapeDescriptor.m_TargetShape << " compared to " << m_VisitorDescriptor.m_TargetShape);
    }

    void VisitReshapeLayer(const IConnectableLayer* layer,
                           const ReshapeDescriptor& reshapeDescriptor,
                           const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckDescriptor(reshapeDescriptor);
        CheckLayerName(name);
    };
};

class TestSpaceToBatchNdLayerVisitor : public TestLayerVisitor
{
private:
    SpaceToBatchNdDescriptor m_VisitorDescriptor;

public:
    explicit TestSpaceToBatchNdLayerVisitor(const SpaceToBatchNdDescriptor& desc, const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_VisitorDescriptor(desc.m_BlockShape, desc.m_PadList)
    {
        m_VisitorDescriptor.m_DataLayout = desc.m_DataLayout;
    };

    void CheckDescriptor(const SpaceToBatchNdDescriptor& desc)
    {
        if (desc.m_BlockShape.size() == m_VisitorDescriptor.m_BlockShape.size())
        {
            for (unsigned int i = 0; i < desc.m_BlockShape.size(); ++i)
            {
                BOOST_CHECK_EQUAL(desc.m_BlockShape[i], m_VisitorDescriptor.m_BlockShape[i]);
            }
        }
        else
        {
            BOOST_ERROR("Unequal vector size for SpaceToBatchNdDescriptor m_BlockShape.");
        }

        if (desc.m_PadList.size() == m_VisitorDescriptor.m_PadList.size())
        {
            for (unsigned int i = 0; i < desc.m_PadList.size(); ++i)
            {
                BOOST_CHECK_EQUAL(desc.m_PadList[i].first, m_VisitorDescriptor.m_PadList[i].first);
                BOOST_CHECK_EQUAL(desc.m_PadList[i].second, m_VisitorDescriptor.m_PadList[i].second);
            }
        }
        else
        {
            BOOST_ERROR("Unequal vector size for SpaceToBatchNdDescriptor m_PadList.");
        }

        BOOST_CHECK(desc.m_DataLayout == m_VisitorDescriptor.m_DataLayout);
    }

    void VisitSpaceToBatchNdLayer(const IConnectableLayer* layer,
                                  const SpaceToBatchNdDescriptor& desc,
                                  const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckDescriptor(desc);
        CheckLayerName(name);
    };
};

class TestMeanLayerVisitor : public TestLayerVisitor
{
private:
    const MeanDescriptor m_VisitorDescriptor;

public:
    explicit TestMeanLayerVisitor(const MeanDescriptor& meanDescriptor, const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_VisitorDescriptor(meanDescriptor.m_Axis, meanDescriptor.m_KeepDims)
    {};

    void CheckDescriptor(const MeanDescriptor& meanDescriptor)
    {
        if (meanDescriptor.m_Axis.size() == m_VisitorDescriptor.m_Axis.size())
        {
            for (unsigned int i = 0; i < meanDescriptor.m_Axis.size(); ++i)
            {
                BOOST_CHECK_EQUAL(meanDescriptor.m_Axis[i], m_VisitorDescriptor.m_Axis[i]);
            }
        }
        else
        {
            BOOST_ERROR("Unequal vector size for MeanDescriptor m_Axis.");
        }

        BOOST_CHECK_EQUAL(meanDescriptor.m_KeepDims, m_VisitorDescriptor.m_KeepDims);
    }

    void VisitMeanLayer(const IConnectableLayer* layer,
                        const MeanDescriptor& meanDescriptor,
                        const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckDescriptor(meanDescriptor);
        CheckLayerName(name);
    };
};

class TestPadLayerVisitor : public TestLayerVisitor
{
private:
    const PadDescriptor m_VisitorDescriptor;

public:
    explicit TestPadLayerVisitor(const PadDescriptor& padDescriptor, const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_VisitorDescriptor(padDescriptor.m_PadList)
    {};

    void CheckDescriptor(const PadDescriptor& padDescriptor)
    {
        if (padDescriptor.m_PadList.size() == m_VisitorDescriptor.m_PadList.size())
        {
            for (unsigned int i = 0; i < padDescriptor.m_PadList.size(); ++i)
            {
                BOOST_CHECK_EQUAL(padDescriptor.m_PadList[i].first, m_VisitorDescriptor.m_PadList[i].first);
                BOOST_CHECK_EQUAL(padDescriptor.m_PadList[i].second, m_VisitorDescriptor.m_PadList[i].second);
            }
        }
        else
        {
            BOOST_ERROR("Unequal vector size for SpaceToBatchNdDescriptor m_PadList.");
        }
    }

    void VisitPadLayer(const IConnectableLayer* layer,
                       const PadDescriptor& padDescriptor,
                       const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckDescriptor(padDescriptor);
        CheckLayerName(name);
    };
};

class TestStridedSliceLayerVisitor : public TestLayerVisitor
{
private:
    StridedSliceDescriptor m_VisitorDescriptor;

public:
    explicit TestStridedSliceLayerVisitor(const StridedSliceDescriptor& desc, const char* name = nullptr)
        : TestLayerVisitor(name)
        , m_VisitorDescriptor(desc.m_Begin, desc.m_End, desc.m_Stride)
    {
        m_VisitorDescriptor.m_BeginMask      = desc.m_BeginMask;
        m_VisitorDescriptor.m_EndMask        = desc.m_EndMask;
        m_VisitorDescriptor.m_ShrinkAxisMask = desc.m_ShrinkAxisMask;
        m_VisitorDescriptor.m_EllipsisMask   = desc.m_EllipsisMask;
        m_VisitorDescriptor.m_NewAxisMask    = desc.m_NewAxisMask;
        m_VisitorDescriptor.m_DataLayout     = desc.m_DataLayout;
    };

    void CheckDescriptor(const StridedSliceDescriptor& desc)
    {
        if (desc.m_Begin.size() == m_VisitorDescriptor.m_Begin.size())
        {
            for (unsigned int i = 0; i < desc.m_Begin.size(); ++i)
            {
                BOOST_CHECK_EQUAL(desc.m_Begin[i], m_VisitorDescriptor.m_Begin[i]);
            }
        }
        else
        {
            BOOST_ERROR("Unequal vector size for StridedSliceDescriptor m_Begin.");
        }

        if (desc.m_End.size() == m_VisitorDescriptor.m_End.size())
        {
            for (unsigned int i = 0; i < desc.m_End.size(); ++i)
            {
                BOOST_CHECK_EQUAL(desc.m_End[i], m_VisitorDescriptor.m_End[i]);
            }
        }
        else
        {
            BOOST_ERROR("Unequal vector size for StridedSliceDescriptor m_End.");
        }

        if (desc.m_Stride.size() == m_VisitorDescriptor.m_Stride.size())
        {
            for (unsigned int i = 0; i < desc.m_Stride.size(); ++i)
            {
                BOOST_CHECK_EQUAL(desc.m_Stride[i], m_VisitorDescriptor.m_Stride[i]);
            }
        }
        else
        {
            BOOST_ERROR("Unequal vector size for StridedSliceDescriptor m_Stride.");
        }

        BOOST_CHECK_EQUAL(desc.m_BeginMask, m_VisitorDescriptor.m_BeginMask);
        BOOST_CHECK_EQUAL(desc.m_EndMask, m_VisitorDescriptor.m_EndMask);
        BOOST_CHECK_EQUAL(desc.m_ShrinkAxisMask, m_VisitorDescriptor.m_ShrinkAxisMask);
        BOOST_CHECK_EQUAL(desc.m_EllipsisMask, m_VisitorDescriptor.m_EllipsisMask);
        BOOST_CHECK_EQUAL(desc.m_NewAxisMask, m_VisitorDescriptor.m_NewAxisMask);
        BOOST_CHECK(desc.m_DataLayout == m_VisitorDescriptor.m_DataLayout);
    }

    void VisitStridedSliceLayer(const IConnectableLayer* layer,
                                const StridedSliceDescriptor& desc,
                                const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckDescriptor(desc);
        CheckLayerName(name);
    };
};

} //namespace armnn
