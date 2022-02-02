//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ResolveType.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <DataTypeUtils.hpp>
#include <armnnTestUtils/LayerTestResult.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/TensorHelpers.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

namespace
{

template<armnn::DataType ArmnnType,
        std::size_t InputDim,
        std::size_t OutputDim,
        typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, OutputDim> BatchToSpaceNdHelper(
        armnn::IWorkloadFactory &workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::DataLayout& dataLayout,
        const unsigned int *inputShape,
        const std::vector<float> &inputData,
        const std::vector<unsigned int> &blockShape,
        const std::vector<std::pair<unsigned int, unsigned int>> &crops,
        const unsigned int *outputShape,
        const std::vector<float> &outputData,
        float scale = 1.0f,
        int32_t offset = 0)
{
    IgnoreUnused(memoryManager);

    armnn::TensorInfo inputTensorInfo(InputDim, inputShape, ArmnnType);
    armnn::TensorInfo outputTensorInfo(OutputDim, outputShape, ArmnnType);

    inputTensorInfo.SetQuantizationScale(scale);
    inputTensorInfo.SetQuantizationOffset(offset);

    outputTensorInfo.SetQuantizationScale(scale);
    outputTensorInfo.SetQuantizationOffset(offset);

    std::vector<T> input = ConvertToDataType<ArmnnType>(inputData, inputTensorInfo);

    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());
    std::vector<T> expectedOutput = ConvertToDataType<ArmnnType>(outputData, outputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::BatchToSpaceNdQueueDescriptor data;
    data.m_Parameters.m_DataLayout = dataLayout;
    data.m_Parameters.m_BlockShape = blockShape;
    data.m_Parameters.m_Crops = crops;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::BatchToSpaceNd,
                                                                                data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, OutputDim>(actualOutput,
                                         expectedOutput,
                                         outputHandle->GetShape(),
                                         outputTensorInfo.GetShape());
}

} // anonymous namespace

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNhwcTest1(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = {4, 2, 2, 1};
    const unsigned int outputShape[] = {1, 4, 4, 1};

    std::vector<float> input({
                                     // Batch 0, Height 0, Width (2) x Channel (1)
                                     1.0f, 3.0f,
                                     // Batch 0, Height 1, Width (2) x Channel (1)
                                     9.0f, 11.0f,


                                     // Batch 1, Height 0, Width (2) x Channel (1)
                                     2.0f, 4.0f,
                                     // Batch 1, Height 1, Width (2) x Channel (1)
                                     10.0f, 12.0f,


                                     // Batch 2, Height 0, Width (2) x Channel (1)
                                     5.0f, 7.0f,
                                     // Batch 2, Height 1, Width (2) x Channel (1)
                                     13.0f, 15.0f,

                                     // Batch 3, Height 0, Width (2) x Channel (3)
                                     6.0f, 8.0f,
                                     // Batch 3, Height 1, Width (2) x Channel (1)
                                     14.0f, 16.0f
                             });

    std::vector<float> expectedOutput({
                                              1.0f,   2.0f,  3.0f,  4.0f,
                                              5.0f,   6.0f,  7.0f,  8.0f,
                                              9.0f,  10.0f, 11.0f,  12.0f,
                                              13.0f, 14.0f, 15.0f,  16.0f
                                      });

    std::vector<unsigned int> blockShape {2, 2};
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager, tensorHandleFactory,
                                                                armnn::DataLayout::NHWC, inputShape, input, blockShape,
                                                                crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNhwcTest2(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = {4, 1, 1, 1};
    const unsigned int outputShape[] = {1, 2, 2, 1};

    std::vector<float> input({
                                     // Batch 0, Height 0, Width (2) x Channel (1)
                                     1.0f, 2.0f, 3.0f, 4.0f
                             });

    std::vector<float> expectedOutput({1.0f, 2.0f, 3.0f, 4.0f});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager, tensorHandleFactory,
                                                                armnn::DataLayout::NHWC, inputShape, input, blockShape,
                                                                crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNhwcTest3(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = {4, 1, 1, 3};
    const unsigned int outputShape[] = {1, 2, 2, 3};

    std::vector<float> input({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    std::vector<float> expectedOutput({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager, tensorHandleFactory,
                                                                armnn::DataLayout::NHWC, inputShape, input, blockShape,
                                                                crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNhwcTest4(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = {8, 1, 3, 1};
    const unsigned int outputShape[] = {2, 2, 4, 1};

    std::vector<float> input({
                                     0.0f, 1.0f, 3.0f,
                                     0.0f, 9.0f, 11.0f,
                                     0.0f, 2.0f, 4.0f,
                                     0.0f, 10.0f, 12.0f,
                                     0.0f, 5.0f, 7.0f,
                                     0.0f, 13.0f, 15.0f,
                                     0.0f, 6.0f, 8.0f,
                                     0.0f, 14.0f, 16.0f
                             });

    std::vector<float> expectedOutput({
                                              1.0f, 2.0f, 3.0f, 4.0f,
                                              5.0f, 6.0f, 7.0f, 8.0f,
                                              9.0f, 10.0f, 11.0f, 12.0f,
                                              13.0f, 14.0f, 15.0f, 16.0f
                                      });

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {2, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager, tensorHandleFactory,
                                                                armnn::DataLayout::NHWC, inputShape, input, blockShape,
                                                                crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNhwcTest5(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = {4, 2, 2, 1};
    const unsigned int outputShape[] = {1, 4, 4, 1};

    std::vector<float> input({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    std::vector<float> expectedOutput({1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager, tensorHandleFactory,
                                                 armnn::DataLayout::NHWC, inputShape,
                                                 input, blockShape, crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNhwcTest6(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = {4, 1, 1, 1};
    const unsigned int outputShape[] = {1, 2, 2, 1};

    std::vector<float> input({
                                     // Batch 0, Height 0, Width (2) x Channel (1)
                                     1, 2, 3, 4
                             });

    std::vector<float> expectedOutput({1, 2, 3, 4});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager, tensorHandleFactory,
                                                 armnn::DataLayout::NHWC, inputShape, input, blockShape,
                                                 crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNhwcTest7(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = {4, 1, 1, 3};
    const unsigned int outputShape[] = {1, 2, 2, 3};

    std::vector<float> input({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    std::vector<float> expectedOutput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager, tensorHandleFactory,
                                                 armnn::DataLayout::NHWC, inputShape, input, blockShape,
                                                 crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNchwTest1(
        armnn::IWorkloadFactory &workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = {4, 3, 1, 1};
    const unsigned int outputShape[] = {1, 3, 2, 2};

    std::vector<float> input({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    std::vector<float> expectedOutput({
                                              // Batch 0, Channel 0, Height (2) x Width (2)
                                              1.0f,  4.0f,
                                              7.0f, 10.0f,

                                              // Batch 0, Channel 1, Height (2) x Width (2)
                                              2.0f,  5.0f,
                                              8.0f, 11.0f,

                                              // Batch 0, Channel 2, Height (2) x Width (2)
                                              3.0f,  6.0f,
                                              9.0f, 12.0f,
                                      });

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager, tensorHandleFactory,
                                                                armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                                                crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNchwTest2(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = {4, 1, 1, 1};
    const unsigned int outputShape[] = {1, 1, 2, 2};

    std::vector<float> input({
                                     // Batch 0, Height 0, Width (2) x Channel (1)
                                     1.0f, 2.0f, 3.0f, 4.0f
                             });

    std::vector<float> expectedOutput({1.0f, 2.0f, 3.0f, 4.0f});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager, tensorHandleFactory,
                                                                armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                                                crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNchwTest3(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = {4, 3, 1, 1};
    const unsigned int outputShape[] = {1, 3, 2, 2};

    std::vector<float> input({1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f});

    std::vector<float> expectedOutput({
                                              // Batch 0, Channel 0, Height (2) x Width (2)
                                              1.0f,  7.0f,
                                              2.0f,  8.0f,

                                              // Batch 0, Channel 1, Height (2) x Width (2)
                                              3.0f,  9.0f,
                                              4.0f, 10.0f,

                                              // Batch 0, Channel 2, Height (2) x Width (2)
                                              5.0f, 11.0f,
                                              6.0f, 12.0f,
                                      });

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager, tensorHandleFactory,
                                                                armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                                                crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNchwTest4(
        armnn::IWorkloadFactory &workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = {4, 3, 1, 1};
    const unsigned int outputShape[] = {1, 3, 2, 2};

    std::vector<float> input({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    std::vector<float> expectedOutput({
                                              // Batch 0, Channel 0, Height (2) x Width (2)
                                              1,  4,
                                              7, 10,

                                              // Batch 0, Channel 1, Height (2) x Width (2)
                                              2,  5,
                                              8, 11,

                                              // Batch 0, Channel 2, Height (2) x Width (2)
                                              3,  6,
                                              9, 12,
                                      });

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager, tensorHandleFactory,
                                                 armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                                 crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNchwTest5(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = {4, 1, 1, 1};
    const unsigned int outputShape[] = {1, 1, 2, 2};

    std::vector<float> input({
                                     // Batch 0, Height 0, Width (2) x Channel (1)
                                     1, 2, 3, 4
                             });

    std::vector<float> expectedOutput({1, 2, 3, 4});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager, tensorHandleFactory,
                                                 armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                                 crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNchwTest6(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = {4, 3, 1, 1};
    const unsigned int outputShape[] = {1, 3, 2, 2};

    std::vector<float> input({1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12});

    std::vector<float> expectedOutput({
                                              // Batch 0, Channel 0, Height (2) x Width (2)
                                              1,  7,
                                              2,  8,

                                              // Batch 0, Channel 1, Height (2) x Width (2)
                                              3,  9,
                                              4, 10,

                                              // Batch 0, Channel 2, Height (2) x Width (2)
                                              5, 11,
                                              6, 12,
                                      });

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager, tensorHandleFactory,
                                                 armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                                 crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNchwTest7(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = {8, 1, 1, 3};
    const unsigned int outputShape[] = {2, 1, 2, 4};

    std::vector<float> input({
                                     0, 1, 3, 0,  9, 11,
                                     0, 2, 4, 0, 10, 12,
                                     0, 5, 7, 0, 13, 15,
                                     0, 6, 8, 0, 14, 16
                             });

    std::vector<float> expectedOutput({
                                              1,  2,  3,  4,
                                              5,  6,  7,  8,
                                              9, 10, 11, 12,
                                              13, 14, 15, 16
                                      });

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {2, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager, tensorHandleFactory,
                                                 armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                                 crops, outputShape, expectedOutput);
}
