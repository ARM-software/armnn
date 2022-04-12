//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GatherNdTestImpl.hpp"

#include <DataTypeUtils.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

namespace
{

template<armnn::DataType ArmnnType,
        typename T = armnn::ResolveType<ArmnnType>,
        size_t ParamsDim,
        size_t IndicesDim,
        size_t OutputDim>
LayerTestResult<T, OutputDim> GatherNdTestImpl(
        armnn::IWorkloadFactory &workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
        const armnn::ITensorHandleFactory &tensorHandleFactory,
        const armnn::TensorInfo &paramsInfo,
        const armnn::TensorInfo &indicesInfo,
        const armnn::TensorInfo &outputInfo,
        const std::vector<T> &paramsData,
        const std::vector<int32_t> &indicesData,
        const std::vector<T> &outputData)
{
    IgnoreUnused(memoryManager);

    std::vector<T> actualOutput(outputInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> paramsHandle = tensorHandleFactory.CreateTensorHandle(paramsInfo);
    std::unique_ptr<armnn::ITensorHandle> indicesHandle = tensorHandleFactory.CreateTensorHandle(indicesInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputInfo);

    armnn::GatherNdQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, paramsInfo, paramsHandle.get());
    AddInputToWorkload(data, info, indicesInfo, indicesHandle.get());
    AddOutputToWorkload(data, info, outputInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::GatherNd,
                                                                                data,
                                                                                info);

    paramsHandle->Allocate();
    indicesHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(paramsHandle.get(), paramsData.data());
    CopyDataToITensorHandle(indicesHandle.get(), indicesData.data());

    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, OutputDim>(actualOutput,
                                         outputData,
                                         outputHandle->GetShape(),
                                         outputInfo.GetShape());
}
} // anonymous namespace

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> SimpleGatherNd2dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo paramsInfo({ 5, 2 }, ArmnnType);
    armnn::TensorInfo indicesInfo({ 3, 1 }, armnn::DataType::Signed32);
    armnn::TensorInfo outputInfo({ 3, 2 }, ArmnnType);
    if (armnn::IsQuantizedType<T>())
    {
        paramsInfo.SetQuantizationScale(1.0f);
        paramsInfo.SetQuantizationOffset(1);
        outputInfo.SetQuantizationScale(1.0f);
        outputInfo.SetQuantizationOffset(1);
    }
    const std::vector<T> params = ConvertToDataType<ArmnnType>(
            { 1, 2,
              3, 4,
              5, 6,
              7, 8,
              9, 10},
            paramsInfo);
    const std::vector<int32_t> indices  = ConvertToDataType<armnn::DataType::Signed32>(
            { 1, 0, 4},
            indicesInfo);
    const std::vector<T> expectedOutput = ConvertToDataType<ArmnnType>(
            { 3, 4,
              1, 2,
              9, 10},
            outputInfo);
    return GatherNdTestImpl<ArmnnType, T, 2, 2, 2>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            paramsInfo,
            indicesInfo,
            outputInfo,
            params,
            indices,
            expectedOutput);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> SimpleGatherNd3dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo paramsInfo({ 2, 3, 8, 4 }, ArmnnType);
    armnn::TensorInfo indicesInfo({ 2, 2 }, armnn::DataType::Signed32);
    armnn::TensorInfo outputInfo({ 2, 8, 4 }, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        paramsInfo.SetQuantizationScale(1.0f);
        paramsInfo.SetQuantizationOffset(0);
        outputInfo.SetQuantizationScale(1.0f);
        outputInfo.SetQuantizationOffset(0);
    }
    const std::vector<T> params = ConvertToDataType<ArmnnType>(
            { 0,   1,   2,   3, 4,   5,   6,   7, 8,   9,  10,  11, 12,  13,  14,  15,
             16,  17,  18,  19, 20,  21,  22,  23, 24,  25,  26,  27, 28,  29,  30,  31,

             32,  33,  34,  35, 36,  37,  38,  39, 40,  41,  42,  43, 44,  45,  46,  47,
             48,  49,  50,  51, 52,  53,  54,  55, 56,  57,  58,  59, 60,  61,  62,  63,

             64,  65,  66,  67, 68,  69,  70,  71, 72,  73,  74,  75, 76,  77,  78,  79,
             80,  81,  82,  83, 84,  85,  86,  87, 88,  89,  90,  91, 92,  93,  94,  95,

             96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,

            128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
            144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,

            160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
            176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191 },
            paramsInfo);

    const std::vector<int32_t> indices  = ConvertToDataType<armnn::DataType::Signed32>(
            { 1, 2, 1, 1},
            indicesInfo);

    const std::vector<T> expectedOutput = ConvertToDataType<ArmnnType>(
            { 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
            176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,

            128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
            144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159},
            outputInfo);

    return GatherNdTestImpl<ArmnnType, T, 4, 2, 3>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            paramsInfo,
            indicesInfo,
            outputInfo,
            params,
            indices,
            expectedOutput);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> SimpleGatherNd4dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo paramsInfo({ 5, 5, 2 }, ArmnnType);
    armnn::TensorInfo indicesInfo({ 2, 2, 3, 2 }, armnn::DataType::Signed32);
    armnn::TensorInfo outputInfo({ 2, 2, 3, 2 }, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        paramsInfo.SetQuantizationScale(1.0f);
        paramsInfo.SetQuantizationOffset(0);
        outputInfo.SetQuantizationScale(1.0f);
        outputInfo.SetQuantizationOffset(0);
    }
    const std::vector<T> params = ConvertToDataType<ArmnnType>(
        { 0,  1,    2,  3,    4,  5,    6,  7,    8,  9,
         10, 11,   12,  13,   14, 15,   16, 17,   18, 19,
         20, 21,   22,  23,   24, 25,   26, 27,   28, 29,
         30, 31,   32,  33,   34, 35,   36, 37,   38, 39,
         40, 41,   42,  43,   44, 45,   46, 47,   48, 49 },
        paramsInfo);

    const std::vector<int32_t> indices  = ConvertToDataType<armnn::DataType::Signed32>(
        { 0, 0,
          3, 3,
          4, 4,

          0, 0,
          1, 1,
          2, 2,

          4, 4,
          3, 3,
          0, 0,

          2, 2,
          1, 1,
          0, 0 },
        indicesInfo);

    const std::vector<T> expectedOutput = ConvertToDataType<ArmnnType>(
        {  0,  1,
          36, 37,
          48, 49,

           0,  1,
          12, 13,
          24, 25,

          48, 49,
          36, 37,
           0,  1,

          24, 25,
          12, 13,
           0,  1 },
        outputInfo);

    return GatherNdTestImpl<ArmnnType, T, 3, 4, 4>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            paramsInfo,
            indicesInfo,
            outputInfo,
            params,
            indices,
            expectedOutput);
}

//
// Explicit template specializations
//

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
SimpleGatherNd2dTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
SimpleGatherNd3dTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
SimpleGatherNd4dTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 2>
SimpleGatherNd2dTest<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 3>
SimpleGatherNd3dTest<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
SimpleGatherNd4dTest<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Signed32>, 2>
SimpleGatherNd2dTest<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Signed32>, 3>
SimpleGatherNd3dTest<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Signed32>, 4>
SimpleGatherNd4dTest<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);