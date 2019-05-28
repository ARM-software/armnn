//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "WorkloadTestUtils.hpp"

#include <armnn/ArmNN.hpp>
#include <armnn/Tensor.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/WorkloadFactory.hpp>
#include <backendsCommon/test/QuantizeHelper.hpp>

#include <test/TensorHelpers.hpp>

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
std::vector<LayerTestResult<T,3>> SplitterTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale = 0.0f,
    int32_t qOffset = 0)
{
    unsigned int inputWidth = 5;
    unsigned int inputHeight = 6;
    unsigned int inputChannels = 3;

    // NOTE: Compute Library imposes a restriction that the x and y dimension (input height and width)
    //       cannot be split.
    //       For the reasons for this, see first comment on https://jira.arm.com/browse/IVGCVSW-1239
    //
    // This test has therefore been recast to split the channels, then split the resulting subtensor.

    // To take channel 0 of original output
    // and channel 0 and channel 1 of the split subtensor.
    unsigned int outputWidth1 = inputWidth;
    unsigned int outputHeight1 = inputHeight;
    unsigned int outputChannels1 = 1;

    // To take channel 1 and 2 of the original output.
    unsigned int outputWidth2 = inputWidth;
    unsigned int outputHeight2 = inputHeight;
    unsigned int outputChannels2 = 2;


    // Define the tensor descriptors.
    armnn::TensorInfo inputTensorInfo({ inputChannels, inputHeight, inputWidth }, ArmnnType, qScale, qOffset);

    // Outputs of the original split.
    armnn::TensorInfo outputTensorInfo1({ outputChannels1, outputHeight1, outputWidth1 }, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputTensorInfo2({ outputChannels2, outputHeight2, outputWidth2 }, ArmnnType, qScale, qOffset);

    // Outputs of the subsequent subtensor split.
    armnn::TensorInfo outputTensorInfo3({ outputChannels1, outputHeight1, outputWidth1 }, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputTensorInfo4({ outputChannels1, outputHeight1, outputWidth1 }, ArmnnType, qScale, qOffset);

    // Set quantization parameters if the requested type is a quantized type.
    // The quantization doesn't really matter as the splitter operator doesn't dequantize/quantize.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo1.SetQuantizationScale(qScale);
        outputTensorInfo1.SetQuantizationOffset(qOffset);
        outputTensorInfo2.SetQuantizationScale(qScale);
        outputTensorInfo2.SetQuantizationOffset(qOffset);
        outputTensorInfo3.SetQuantizationScale(qScale);
        outputTensorInfo3.SetQuantizationOffset(qOffset);
        outputTensorInfo4.SetQuantizationScale(qScale);
        outputTensorInfo4.SetQuantizationOffset(qOffset);
    }

    LayerTestResult<T,3> ret1(outputTensorInfo1);
    LayerTestResult<T,3> ret2(outputTensorInfo2);
    LayerTestResult<T,3> ret3(outputTensorInfo3);
    LayerTestResult<T,3> ret4(outputTensorInfo4);

    auto input = MakeTensor<T, 3>(inputTensorInfo, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
            11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
            16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f, 24.0f, 25.0f,
            26.0f, 27.0f, 28.0f, 29.0f, 30.0f,

            31.0f, 32.0f, 33.0f, 34.0f, 35.0f,
            36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
            41.0f, 42.0f, 43.0f, 44.0f, 45.0f,
            46.0f, 47.0f, 48.0f, 49.0f, 50.0f,
            51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
            56.0f, 57.0f, 58.0f, 59.0f, 60.0f,

            61.0f, 62.0f, 63.0f, 64.0f, 65.0f,
            66.0f, 67.0f, 68.0f, 69.0f, 70.0f,
            71.0f, 72.0f, 73.0f, 74.0f, 75.0f,
            76.0f, 77.0f, 78.0f, 79.0f, 80.0f,
            81.0f, 82.0f, 83.0f, 84.0f, 85.0f,
            86.0f, 87.0f, 88.0f, 89.0f, 90.0f,
        })
    ));

    // Channel 0 of the original input.
    ret1.outputExpected = MakeTensor<T, 3>(outputTensorInfo1, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
            11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
            16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f, 24.0f, 25.0f,
            26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
        })
    ));

    // Channel 1 & 2 of the original input.
    ret2.outputExpected = MakeTensor<T, 3>(outputTensorInfo2, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            31.0f, 32.0f, 33.0f, 34.0f, 35.0f,
            36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
            41.0f, 42.0f, 43.0f, 44.0f, 45.0f,
            46.0f, 47.0f, 48.0f, 49.0f, 50.0f,
            51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
            56.0f, 57.0f, 58.0f, 59.0f, 60.0f,

            61.0f, 62.0f, 63.0f, 64.0f, 65.0f,
            66.0f, 67.0f, 68.0f, 69.0f, 70.0f,
            71.0f, 72.0f, 73.0f, 74.0f, 75.0f,
            76.0f, 77.0f, 78.0f, 79.0f, 80.0f,
            81.0f, 82.0f, 83.0f, 84.0f, 85.0f,
            86.0f, 87.0f, 88.0f, 89.0f, 90.0f,
        })
    ));

    // Channel 0 of return 2 (i.e. channels 1 and 2 of the original input).
    ret3.outputExpected = MakeTensor<T, 3>(outputTensorInfo3, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            31.0f, 32.0f, 33.0f, 34.0f, 35.0f,
            36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
            41.0f, 42.0f, 43.0f, 44.0f, 45.0f,
            46.0f, 47.0f, 48.0f, 49.0f, 50.0f,
            51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
            56.0f, 57.0f, 58.0f, 59.0f, 60.0f,
        })
    ));

    // Channel 1 of return 2.
    ret4.outputExpected = MakeTensor<T, 3>(outputTensorInfo4, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            61.0f, 62.0f, 63.0f, 64.0f, 65.0f,
            66.0f, 67.0f, 68.0f, 69.0f, 70.0f,
            71.0f, 72.0f, 73.0f, 74.0f, 75.0f,
            76.0f, 77.0f, 78.0f, 79.0f, 80.0f,
            81.0f, 82.0f, 83.0f, 84.0f, 85.0f,
            86.0f, 87.0f, 88.0f, 89.0f, 90.0f,
        })
    ));

    // NOTE: as a corollary of the splitting of x and y restriction the x and y values of the view origins
    //       have to be zero, the co-ordinates are as per the tensor info above channels, height/y, width/x
    //       note that under the hood the compute engine reverses these i.e. its coordinate system is x, y, channels.
    std::vector<unsigned int> wOrigin1 = {0, 0, 0}; //Extent of the window is defined by size of output[0].
    armnn::SplitterQueueDescriptor::ViewOrigin window1(wOrigin1);

    std::vector<unsigned int> wOrigin2 = {1, 0, 0}; //Extent of the window is defined by size of output[1].
    armnn::SplitterQueueDescriptor::ViewOrigin window2(wOrigin2);

    std::vector<unsigned int> wOrigin3 = {0, 0, 0}; //Extent of the window is defined by size of output[2].
    armnn::SplitterQueueDescriptor::ViewOrigin window3(wOrigin3);

    std::vector<unsigned int> wOrigin4 = {1, 0, 0}; //Extent of the window is defined by size of output[3].
    armnn::SplitterQueueDescriptor::ViewOrigin window4(wOrigin4);

    bool subTensorsSupported = workloadFactory.SupportsSubTensors();

    std::unique_ptr<armnn::ITensorHandle> inputHandle  = workloadFactory.CreateTensorHandle(inputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> outputHandle1 =
        subTensorsSupported ?
            workloadFactory.CreateSubTensorHandle(*inputHandle, outputTensorInfo1.GetShape(), wOrigin1.data()) :
            workloadFactory.CreateTensorHandle(outputTensorInfo1);

    std::unique_ptr<armnn::ITensorHandle> outputHandle2 =
        subTensorsSupported ?
            workloadFactory.CreateSubTensorHandle(*inputHandle, outputTensorInfo2.GetShape(), wOrigin2.data()) :
            workloadFactory.CreateTensorHandle(outputTensorInfo2);

    std::unique_ptr<armnn::ITensorHandle> outputHandle3 =
        subTensorsSupported ?
            workloadFactory.CreateSubTensorHandle(*outputHandle2, outputTensorInfo3.GetShape(), wOrigin3.data()) :
            workloadFactory.CreateTensorHandle(outputTensorInfo3);

    std::unique_ptr<armnn::ITensorHandle> outputHandle4 =
        subTensorsSupported ?
            workloadFactory.CreateSubTensorHandle(*outputHandle2, outputTensorInfo4.GetShape(), wOrigin4.data()) :
            workloadFactory.CreateTensorHandle(outputTensorInfo4);

    // Do the first split
    armnn::SplitterQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo1, outputHandle1.get());
    AddOutputToWorkload(data, info, outputTensorInfo2, outputHandle2.get());

    data.m_ViewOrigins.push_back(window1);
    data.m_ViewOrigins.push_back(window2);

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateSplitter(data, info);

    inputHandle->Allocate();
    outputHandle1->Allocate();
    outputHandle2->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&ret1.output[0][0][0], outputHandle1.get());
    CopyDataFromITensorHandle(&ret2.output[0][0][0], outputHandle2.get());

    // Do the second split.
    armnn::SplitterQueueDescriptor data2;
    armnn::WorkloadInfo info2;
    AddInputToWorkload(data2, info2, outputTensorInfo2, outputHandle2.get());
    AddOutputToWorkload(data2, info2, outputTensorInfo3, outputHandle3.get());
    AddOutputToWorkload(data2, info2, outputTensorInfo4, outputHandle4.get());

    data2.m_ViewOrigins.push_back(window3);
    data2.m_ViewOrigins.push_back(window4);

    std::unique_ptr<armnn::IWorkload> workload2 = workloadFactory.CreateSplitter(data2, info2);

    outputHandle3->Allocate();
    outputHandle4->Allocate();

    ExecuteWorkload(*workload2, memoryManager);

    CopyDataFromITensorHandle(&ret3.output[0][0][0], outputHandle3.get());
    CopyDataFromITensorHandle(&ret4.output[0][0][0], outputHandle4.get());

    std::vector<LayerTestResult<T,3>> ret = {ret1, ret2, ret3, ret4,};

    return ret;
}


template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> CopyViaSplitterTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale, int32_t qOffset)
{
    const armnn::TensorInfo tensorInfo({ 3, 6, 5 }, ArmnnType, qScale, qOffset);
    auto input = MakeTensor<T, 3>(tensorInfo, QuantizedVector<T>(qScale, qOffset,
                                                                 {
                                                                     1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                                                     6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                                                     11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                                                                     16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                                                                     21.0f, 22.0f, 23.0f, 24.0f, 25.0f,
                                                                     26.0f, 27.0f, 28.0f, 29.0f, 30.0f,

                                                                     31.0f, 32.0f, 33.0f, 34.0f, 35.0f,
                                                                     36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
                                                                     41.0f, 42.0f, 43.0f, 44.0f, 45.0f,
                                                                     46.0f, 47.0f, 48.0f, 49.0f, 50.0f,
                                                                     51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
                                                                     56.0f, 57.0f, 58.0f, 59.0f, 60.0f,

                                                                     61.0f, 62.0f, 63.0f, 64.0f, 65.0f,
                                                                     66.0f, 67.0f, 68.0f, 69.0f, 70.0f,
                                                                     71.0f, 72.0f, 73.0f, 74.0f, 75.0f,
                                                                     76.0f, 77.0f, 78.0f, 79.0f, 80.0f,
                                                                     81.0f, 82.0f, 83.0f, 84.0f, 85.0f,
                                                                     86.0f, 87.0f, 88.0f, 89.0f, 90.0f,
                                                                 }));

    std::vector<unsigned int> origin = { 0, 0, 0 };
    armnn::SplitterQueueDescriptor::ViewOrigin window(origin);

    const bool subTensorsSupported = workloadFactory.SupportsSubTensors();

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(tensorInfo);

    std::unique_ptr<armnn::ITensorHandle> outputHandle =
        subTensorsSupported ?
            workloadFactory.CreateSubTensorHandle(*inputHandle, tensorInfo.GetShape(), origin.data()) :
            workloadFactory.CreateTensorHandle(tensorInfo);

    armnn::SplitterQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, tensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, tensorInfo, outputHandle.get());

    data.m_ViewOrigins.push_back(window);

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateSplitter(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0]);

    workload->Execute();

    LayerTestResult<T, 3> ret(tensorInfo);
    CopyDataFromITensorHandle(&ret.output[0][0][0], outputHandle.get());
    ret.outputExpected = input;

    return ret;
}
