//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ResizeTestImpl.hpp"

#include <armnnUtils/QuantizeHelper.hpp>


#include <armnnUtils/TensorUtils.hpp>
#include <armnnUtils/DataLayoutIndexed.hpp>
#include <armnnUtils/Permute.hpp>

#include <armnnTestUtils/DataLayoutUtils.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

namespace
{

struct ResizeTestParams
{
    ResizeTestParams()
        : m_ResizeMethod(armnn::ResizeMethod::Bilinear)
        , m_DataLayout(armnn::DataLayout::NCHW)
        , m_InQuantScale(1.0f)
        , m_InQuantOffset(0)
        , m_OutQuantScale(1.0f)
        , m_OutQuantOffset(0)
        , m_AlignCorners(false)
        , m_HalfPixelCenters(false) {}

    armnn::ResizeMethod m_ResizeMethod;
    armnn::DataLayout   m_DataLayout;

    armnn::TensorShape  m_InputShape;
    armnn::TensorShape  m_OutputShape;

    std::vector<float>  m_InputData;
    std::vector<float>  m_ExpectedOutputData;

    float               m_InQuantScale;
    int32_t             m_InQuantOffset;

    float               m_OutQuantScale;
    int32_t             m_OutQuantOffset;

    bool m_AlignCorners;
    bool m_HalfPixelCenters;

    void SetInQuantParams(float quantScale, int32_t quantOffset)
    {
        m_InQuantScale   = quantScale;
        m_InQuantOffset  = quantOffset;
    }

    void SetOutQuantParams(float quantScale, int32_t quantOffset)
    {
        m_OutQuantScale  = quantScale;
        m_OutQuantOffset = quantOffset;
    }

    void SetInOutQuantParams(float quantScale, int32_t quantOffset)
    {
        SetInQuantParams(quantScale, quantOffset);
        SetOutQuantParams(quantScale, quantOffset);
    }
};

template<size_t NumDims,
         armnn::DataType ArmnnType,
         typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, NumDims> ResizeTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const ResizeTestParams& params)
{
    IgnoreUnused(memoryManager);
    armnn::TensorInfo inputInfo(params.m_InputShape, ArmnnType);
    armnn::TensorInfo outputInfo(params.m_OutputShape, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputInfo.SetQuantizationScale(params.m_InQuantScale);
        inputInfo.SetQuantizationOffset(params.m_InQuantOffset);

        outputInfo.SetQuantizationScale(params.m_OutQuantScale);
        outputInfo.SetQuantizationOffset(params.m_OutQuantOffset);
    }

    std::vector<T> inputData =
        armnnUtils::QuantizedVector<T>(params.m_InputData, params.m_InQuantScale, params.m_InQuantOffset);

    std::vector<T> actualOutput(outputInfo.GetNumElements());
    std::vector<T> expectedOutputData = armnnUtils::QuantizedVector<T>(params.m_ExpectedOutputData,
                                                                       params.m_OutQuantScale,
                                                                       params.m_OutQuantOffset);

    if (params.m_DataLayout == armnn::DataLayout::NHWC)
    {
        PermuteTensorNchwToNhwc(inputInfo, inputData);
        PermuteTensorNchwToNhwc(outputInfo, expectedOutputData);
    }

    std::unique_ptr<armnn::ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Method     = params.m_ResizeMethod;
    descriptor.m_Parameters.m_DataLayout = params.m_DataLayout;
    descriptor.m_Parameters.m_AlignCorners = params.m_AlignCorners;
    descriptor.m_Parameters.m_HalfPixelCenters = params.m_HalfPixelCenters;

    armnnUtils::DataLayoutIndexed dataLayoutIndexed(params.m_DataLayout);
    descriptor.m_Parameters.m_TargetWidth  = params.m_OutputShape[dataLayoutIndexed.GetWidthIndex()];
    descriptor.m_Parameters.m_TargetHeight = params.m_OutputShape[dataLayoutIndexed.GetHeightIndex()];

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Resize,
                                                                                descriptor,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), inputData.data());

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, NumDims>(actualOutput,
                                       expectedOutputData,
                                       outputHandle->GetShape(),
                                       outputInfo.GetShape());
}

} // anonymous namespace

//
// Bilinear
//

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ResizeBilinearNopTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    ResizeTestParams testParams;
    testParams.m_ResizeMethod = armnn::ResizeMethod::Bilinear;
    testParams.m_DataLayout   = dataLayout;

    testParams.m_InputShape  = { 1, 2, 4, 4 };
    testParams.m_OutputShape = testParams.m_InputShape;

    testParams.m_InputData =
    {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 4.0f, 5.0f,
        3.0f, 4.0f, 5.0f, 6.0f,
        4.0f, 5.0f, 6.0f, 7.0f,

        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 4.0f, 5.0f,
        3.0f, 4.0f, 5.0f, 6.0f,
        4.0f, 5.0f, 6.0f, 7.0f
    };

    testParams.m_ExpectedOutputData = testParams.m_InputData;

    testParams.SetInOutQuantParams(1.5f, 3);

    return ResizeTestImpl<4, ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory, testParams);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> SimpleResizeBilinearTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    ResizeTestParams testParams;
    testParams.m_ResizeMethod = armnn::ResizeMethod::Bilinear;
    testParams.m_DataLayout   = dataLayout;

    testParams.m_InputShape  = { 1, 2, 2, 2 };
    testParams.m_OutputShape = { 1, 2, 1, 1 };

    testParams.m_InputData =
    {
          1.0f, 255.0f,
        200.0f, 250.0f,

        250.0f, 200.0f,
        250.0f,   1.0f
    };

    // The 'resize' operation projects the top-left corner of output texels into the input image,
    // then figures out the interpolants and weights. Note this is different to projecting the centre of the
    // output texel. Thus, for a input matrix of 2x2, we'll expect the output 1x1 matrix to contain, as
    // its single element, the value that was at position (0,0) of the input matrix (rather than an average,
    // which we would expect if projecting the centre).
    testParams.m_ExpectedOutputData =
    {
          1.0f,

        250.0f
    };

    testParams.SetInOutQuantParams(0.1567f, 1);

    return ResizeTestImpl<4, ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory, testParams);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ResizeBilinearSqMinTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    ResizeTestParams testParams;
    testParams.m_ResizeMethod = armnn::ResizeMethod::Bilinear;
    testParams.m_DataLayout   = dataLayout;

    testParams.m_InputShape  = { 1, 2, 4, 4 };
    testParams.m_OutputShape = { 1, 2, 2, 2 };

    testParams.m_InputData =
    {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 4.0f, 5.0f,
        3.0f, 4.0f, 5.0f, 6.0f,
        4.0f, 5.0f, 6.0f, 7.0f,

        7.0f, 6.0f, 5.0f, 4.0f,
        6.0f, 5.0f, 4.0f, 3.0f,
        5.0f, 4.0f, 3.0f, 2.0f,
        4.0f, 3.0f, 2.0f, 1.0f
    };

    testParams.m_ExpectedOutputData =
    {
        1.0f, 3.0f,
        3.0f, 5.0f,

        7.0f, 5.0f,
        5.0f, 3.0f
    };

    testParams.SetInOutQuantParams(3.141592f, 3);

    return ResizeTestImpl<4, ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory, testParams);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ResizeBilinearMinTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    ResizeTestParams testParams;
    testParams.m_ResizeMethod = armnn::ResizeMethod::Bilinear;
    testParams.m_DataLayout   = dataLayout;

    testParams.m_InputShape  = { 1, 2, 3, 5 };
    testParams.m_OutputShape = { 1, 2, 2, 3 };

    testParams.m_InputData =
    {
         1.5f,  3.0f,  4.5f,  6.0f,  7.5f,
         9.0f, 10.5f, 12.0f, 13.5f, 15.0f,
        16.5f, 18.0f, 19.5f, 21.0f, 22.5f,

        16.5f, 18.0f, 19.5f, 21.0f, 22.5f,
         9.0f, 10.5f, 12.0f, 13.5f, 15.0f,
         1.5f,  3.0f,  4.5f,  6.0f,  7.5f
    };

    testParams.m_ExpectedOutputData =
    {
         1.50f,  4.00f,  6.50f,
        12.75f, 15.25f, 17.75f,

        16.50f, 19.00f, 21.50f,
         5.25f,  7.75f, 10.25f
    };

    testParams.SetInOutQuantParams(1.5f, -1);

    return ResizeTestImpl<4, ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory, testParams);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ResizeBilinearMagTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    ResizeTestParams testParams;
    testParams.m_ResizeMethod = armnn::ResizeMethod::Bilinear;
    testParams.m_DataLayout   = dataLayout;

    testParams.m_InputShape  = { 1, 2, 3, 2 };
    testParams.m_OutputShape = { 1, 2, 3, 5 };

    testParams.m_InputData =
    {
          1.0f,   2.0f,
         13.0f,  21.0f,
        144.0f, 233.0f,

        233.0f, 144.0f,
         21.0f,  13.0f,
          2.0f,   1.0f
    };

    testParams.m_ExpectedOutputData =
    {
          1.0f,   1.4f,   1.8f,   2.0f,   2.0f,
         13.0f,  16.2f,  19.4f,  21.0f,  21.0f,
        144.0f, 179.6f, 215.2f, 233.0f, 233.0f,

        233.0f, 197.4f, 161.8f, 144.0f, 144.0f,
         21.0f,  17.8f,  14.6f,  13.0f,  13.0f,
          2.0f,   1.6f,   1.2f,   1.0f,   1.0f
    };

    testParams.SetInQuantParams(1.0f, 0);

    return ResizeTestImpl<4, ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory, testParams);
}

//
// NearestNeighbor
//

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ResizeNearestNeighborNopTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    ResizeTestParams testParams;
    testParams.m_ResizeMethod = armnn::ResizeMethod::NearestNeighbor;
    testParams.m_DataLayout   = dataLayout;

    testParams.m_InputShape  = { 1, 2, 4, 4 };
    testParams.m_OutputShape = testParams.m_InputShape;

    testParams.m_InputData =
    {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 4.0f, 5.0f,
        3.0f, 4.0f, 5.0f, 6.0f,
        4.0f, 5.0f, 6.0f, 7.0f,

        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 4.0f, 5.0f,
        3.0f, 4.0f, 5.0f, 6.0f,
        4.0f, 5.0f, 6.0f, 7.0f
    };

    testParams.m_ExpectedOutputData = testParams.m_InputData;

    testParams.SetInOutQuantParams(1.5f, 3);

    return ResizeTestImpl<4, ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory, testParams);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> SimpleResizeNearestNeighborTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    ResizeTestParams testParams;
    testParams.m_ResizeMethod = armnn::ResizeMethod::NearestNeighbor;
    testParams.m_DataLayout   = dataLayout;

    testParams.m_InputShape  = { 1, 2, 2, 2 };
    testParams.m_OutputShape = { 1, 2, 1, 1 };

    testParams.m_InputData =
    {
          1.0f, 255.0f,
        200.0f, 250.0f,

        250.0f, 200.0f,
        250.0f,   1.0f
    };

    // The 'resize' operation projects the top-left corner of output texels into the input image,
    // then figures out the interpolants and weights. Note this is different to projecting the centre of the
    // output texel. Thus, for a input matrix of 2x2, we'll expect the output 1x1 matrix to contain, as
    // its single element, the value that was at position (0,0) of the input matrix (rather than an average,
    // which we would expect if projecting the centre).
    testParams.m_ExpectedOutputData =
    {
          1.0f,

        250.0f
    };

    testParams.SetInOutQuantParams(0.1567f, 1);

    return ResizeTestImpl<4, ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory, testParams);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ResizeNearestNeighborSqMinTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    ResizeTestParams testParams;
    testParams.m_ResizeMethod = armnn::ResizeMethod::NearestNeighbor;
    testParams.m_DataLayout   = dataLayout;

    testParams.m_InputShape  = { 1, 2, 4, 4 };
    testParams.m_OutputShape = { 1, 2, 2, 2 };

    testParams.m_InputData =
    {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 4.0f, 5.0f,
        3.0f, 4.0f, 5.0f, 6.0f,
        4.0f, 5.0f, 6.0f, 7.0f,

        7.0f, 6.0f, 5.0f, 4.0f,
        6.0f, 5.0f, 4.0f, 3.0f,
        5.0f, 4.0f, 3.0f, 2.0f,
        4.0f, 3.0f, 2.0f, 1.0f
    };

    testParams.m_ExpectedOutputData =
    {
        1.0f, 3.0f,
        3.0f, 5.0f,

        7.0f, 5.0f,
        5.0f, 3.0f
    };

    testParams.SetInOutQuantParams(3.141592f, 3);

    return ResizeTestImpl<4, ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory, testParams);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ResizeNearestNeighborMinTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
        ResizeTestParams testParams;
    testParams.m_ResizeMethod = armnn::ResizeMethod::NearestNeighbor;
    testParams.m_DataLayout   = dataLayout;

    testParams.m_InputShape  = { 1, 2, 3, 5 };
    testParams.m_OutputShape = { 1, 2, 2, 3 };

    testParams.m_InputData =
    {
         1.5f,  3.0f,  4.5f,  6.0f,  7.5f,
         9.0f, 10.5f, 12.0f, 13.5f, 15.0f,
        16.5f, 18.0f, 19.5f, 21.0f, 22.5f,

        16.5f, 18.0f, 19.5f, 21.0f, 22.5f,
         9.0f, 10.5f, 12.0f, 13.5f, 15.0f,
         1.5f,  3.0f,  4.5f,  6.0f,  7.5f
    };

    testParams.m_ExpectedOutputData =
    {
         1.5f,  3.0f,  6.0f,
         9.0f, 10.5f, 13.5f,

        16.5f, 18.0f, 21.0f,
         9.0f, 10.5f, 13.5f
    };

    testParams.SetInOutQuantParams(1.5f, -1);

    return ResizeTestImpl<4, ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory, testParams);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ResizeNearestNeighborMagTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout,
        float inQuantScale,
        int32_t inQuantOffset,
        float outQuantScale,
        int32_t outQuantOffset)
{
    ResizeTestParams testParams;
    testParams.m_ResizeMethod = armnn::ResizeMethod::NearestNeighbor;
    testParams.m_DataLayout   = dataLayout;

    testParams.m_InputShape  = { 1, 2, 3, 2 };
    testParams.m_OutputShape = { 1, 2, 3, 5 };

    testParams.m_InputData =
    {
        0.183005f, 2.379065f,
        1.054970f, 1.302565f,
        2.400595f, 0.688960f,

        2.400595f, 0.688960f,
        1.054970f, 1.302565f,
        0.183005f, 2.379065f,
    };

    testParams.m_ExpectedOutputData =
    {
        0.183005f, 0.183005f, 0.183005f, 2.379065f, 2.379065f,
        1.054970f, 1.054970f, 1.054970f, 1.302565f, 1.302565f,
        2.400595f, 2.400595f, 2.400595f, 0.688960f, 0.688960f,

        2.400595f, 2.400595f, 2.400595f, 0.688960f, 0.688960f,
        1.054970f, 1.054970f, 1.054970f, 1.302565f, 1.302565f,
        0.183005f, 0.183005f, 0.183005f, 2.379065f, 2.379065f
    };

    testParams.SetInQuantParams(inQuantScale, inQuantOffset);
    testParams.SetOutQuantParams(outQuantScale, outQuantOffset);

    return ResizeTestImpl<4, ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory, testParams);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> HalfPixelCentersResizeBilinearTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    ResizeTestParams testParams;
    testParams.m_ResizeMethod = armnn::ResizeMethod::Bilinear;
    testParams.m_DataLayout   = dataLayout;
    testParams.m_HalfPixelCenters = true;

    testParams.m_InputShape  = { 2, 1, 2, 2 };
    testParams.m_OutputShape = { 2, 1, 3, 3 };

    testParams.m_InputData =
    {
          1.0f, 2.0f,
          3.0f, 4.0f,

          1.0f, 2.0f,
          3.0f, 4.0f
    };

    testParams.m_ExpectedOutputData =
    {
          1.0f, 1.5f, 2.0f,
          2.0f, 2.5f, 3.0f,
          3.0f, 3.5f, 4.0f,

          1.0f, 1.5f, 2.0f,
          2.0f, 2.5f, 3.0f,
          3.0f, 3.5f, 4.0f,
    };

    return ResizeTestImpl<4, ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory, testParams);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> AlignCornersResizeBilinearTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    ResizeTestParams testParams;
    testParams.m_ResizeMethod = armnn::ResizeMethod::Bilinear;
    testParams.m_DataLayout   = dataLayout;
    testParams.m_AlignCorners = true;

    testParams.m_InputShape  = { 1, 1, 2, 2 };
    testParams.m_OutputShape = { 1, 1, 1, 1 };

    testParams.m_InputData =
    {
          1.0f, 2.0f,
          3.0f, 4.0f,
    };

    testParams.m_ExpectedOutputData =
    {
          1.0f
    };

    return ResizeTestImpl<4, ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory, testParams);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> HalfPixelCentersResizeNearestNeighbourTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    ResizeTestParams testParams;
    testParams.m_ResizeMethod = armnn::ResizeMethod::NearestNeighbor;
    testParams.m_DataLayout   = dataLayout;
    testParams.m_HalfPixelCenters = true;

    testParams.m_InputShape  = { 1, 1, 2, 5 };
    testParams.m_OutputShape = { 1, 1, 2, 2 };

    testParams.m_InputData =
    {
          1.0f, 2.0f, 3.0f, 4.0f, 5.0f,

          1.0f, 2.0f, 3.0f, 4.0f, 5.0f
    };

    testParams.m_ExpectedOutputData =
    {
          2.0f, 4.0f,
          2.0f, 4.0f
    };

    return ResizeTestImpl<4, ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory, testParams);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> AlignCornersResizeNearestNeighbourTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout)
{
    ResizeTestParams testParams;
    testParams.m_ResizeMethod = armnn::ResizeMethod::NearestNeighbor;
    testParams.m_DataLayout   = dataLayout;
    testParams.m_AlignCorners = true;

    testParams.m_InputShape  = { 1, 1, 2, 2 };
    testParams.m_OutputShape = { 1, 1, 1, 1 };

    testParams.m_InputData =
    {
          1.0f, 2.0f,
          3.0f, 4.0f,
    };

    testParams.m_ExpectedOutputData =
    {
          1.0f
    };

    return ResizeTestImpl<4, ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory, testParams);
}

//
// Explicit template instantiations
//

// Float32
template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
ResizeBilinearNopTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
SimpleResizeBilinearTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
ResizeBilinearSqMinTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
ResizeBilinearMinTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
ResizeBilinearMagTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
ResizeNearestNeighborNopTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
SimpleResizeNearestNeighborTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
ResizeNearestNeighborSqMinTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
ResizeNearestNeighborMinTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
ResizeNearestNeighborMagTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout,
    float inQuantScale,
    int32_t inQuantOffset,
    float outQuantScale,
    int32_t outQuantOffset);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
HalfPixelCentersResizeBilinearTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
AlignCornersResizeBilinearTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
HalfPixelCentersResizeNearestNeighbourTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
AlignCornersResizeNearestNeighbourTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

// Float16
template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
ResizeBilinearNopTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
SimpleResizeBilinearTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
ResizeBilinearSqMinTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
ResizeBilinearMinTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
ResizeBilinearMagTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
ResizeNearestNeighborNopTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
SimpleResizeNearestNeighborTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
ResizeNearestNeighborSqMinTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
ResizeNearestNeighborMinTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
ResizeNearestNeighborMagTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout,
    float inQuantScale,
    int32_t inQuantOffset,
    float outQuantScale,
    int32_t outQuantOffset);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
HalfPixelCentersResizeBilinearTest<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
AlignCornersResizeBilinearTest<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
HalfPixelCentersResizeNearestNeighbourTest<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
AlignCornersResizeNearestNeighbourTest<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout);

// QAsymm8
template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
ResizeBilinearNopTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
SimpleResizeBilinearTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
ResizeBilinearSqMinTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
ResizeBilinearMinTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
ResizeBilinearMagTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
ResizeNearestNeighborNopTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
SimpleResizeNearestNeighborTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
ResizeNearestNeighborSqMinTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
ResizeNearestNeighborMinTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
ResizeNearestNeighborMagTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout,
    float inQuantScale,
    int32_t inQuantOffset,
    float outQuantScale,
    int32_t outQuantOffset);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
HalfPixelCentersResizeBilinearTest<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
AlignCornersResizeBilinearTest<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
HalfPixelCentersResizeNearestNeighbourTest<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
AlignCornersResizeNearestNeighbourTest<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout);

// QAsymmS8
template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
ResizeBilinearNopTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
SimpleResizeBilinearTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
ResizeBilinearSqMinTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
ResizeBilinearMinTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
ResizeBilinearMagTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
ResizeNearestNeighborNopTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
SimpleResizeNearestNeighborTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
ResizeNearestNeighborSqMinTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
ResizeNearestNeighborMinTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
ResizeNearestNeighborMagTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout,
    float inQuantScale,
    int32_t inQuantOffset,
    float outQuantScale,
    int32_t outQuantOffset);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
HalfPixelCentersResizeBilinearTest<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
AlignCornersResizeBilinearTest<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
HalfPixelCentersResizeNearestNeighbourTest<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
AlignCornersResizeNearestNeighbourTest<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout);

// QSymm16
template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
ResizeBilinearNopTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
SimpleResizeBilinearTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
ResizeBilinearSqMinTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
ResizeBilinearMinTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
ResizeBilinearMagTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
ResizeNearestNeighborNopTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
SimpleResizeNearestNeighborTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
ResizeNearestNeighborSqMinTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
ResizeNearestNeighborMinTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
ResizeNearestNeighborMagTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout dataLayout,
    float inQuantScale,
    int32_t inQuantOffset,
    float outQuantScale,
    int32_t outQuantOffset);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
HalfPixelCentersResizeBilinearTest<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
AlignCornersResizeBilinearTest<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
HalfPixelCentersResizeNearestNeighbourTest<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
AlignCornersResizeNearestNeighbourTest<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout dataLayout);