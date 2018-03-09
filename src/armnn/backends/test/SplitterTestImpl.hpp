//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <armnn/ArmNN.hpp>
#include <armnn/Tensor.hpp>
#include <backends/WorkloadInfo.hpp>

#include "test/TensorHelpers.hpp"

#include "backends/CpuTensorHandle.hpp"
#include "backends/WorkloadFactory.hpp"

#include "backends/test/QuantizeHelper.hpp"


template<typename T>
std::vector<LayerTestResult<T,3>> SplitterTestCommon(armnn::IWorkloadFactory& workloadFactory,
                                                     float qScale = 0.0f,
                                                     int32_t qOffset = 0)
{
    unsigned int inputWidth = 5;
    unsigned int inputHeight = 6;
    unsigned int inputChannels = 3;

    unsigned int outputWidth1 = 2;
    unsigned int outputHeight1 = 2;
    unsigned int outputChannels1 = 3;

    unsigned int outputWidth2 = 2;
    unsigned int outputHeight2 = 4;
    unsigned int outputChannels2 = 3;

    unsigned int outputWidth3 = 3;
    unsigned int outputHeight3 = 6;
    unsigned int outputChannels3 = 2;

    unsigned int outputWidth4 = 3;
    unsigned int outputHeight4 = 6;
    unsigned int outputChannels4 = 1;


    // Define the tensor descriptors
    armnn::TensorInfo inputTensorInfo({ inputChannels, inputHeight, inputWidth }, armnn::GetDataType<T>());
    armnn::TensorInfo outputTensorInfo1({ outputChannels1, outputHeight1, outputWidth1 }, armnn::GetDataType<T>());
    armnn::TensorInfo outputTensorInfo2({ outputChannels2, outputHeight2, outputWidth2 }, armnn::GetDataType<T>());
    armnn::TensorInfo outputTensorInfo3({ outputChannels3, outputHeight3, outputWidth3 }, armnn::GetDataType<T>());
    armnn::TensorInfo outputTensorInfo4({ outputChannels4, outputHeight4, outputWidth4 }, armnn::GetDataType<T>());
    // note that output 5 should match output 2
    armnn::TensorInfo outputTensorInfo5({ outputChannels2, outputHeight2, outputWidth2 }, armnn::GetDataType<T>());

    // Set quantization parameters if the requested type is a quantized type.
    // The quantization doesn't really matter as the splitter operator doesn't dequantize/quantize
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
        outputTensorInfo5.SetQuantizationScale(qScale);
        outputTensorInfo5.SetQuantizationOffset(qOffset);
    }

    LayerTestResult<T,3> ret1(outputTensorInfo1);
    LayerTestResult<T,3> ret2(outputTensorInfo2);
    LayerTestResult<T,3> ret3(outputTensorInfo3);
    LayerTestResult<T,3> ret4(outputTensorInfo4);
    LayerTestResult<T,3> ret5(outputTensorInfo5);

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


    ret1.outputExpected = MakeTensor<T, 3>(outputTensorInfo1, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            1.0f, 2.0f,
            6.0f, 7.0f,

            31.0f, 32.0f,
            36.0f, 37.0f,

            61.0f, 62.0f,
            66.0f, 67.0f,
        })
    ));

    ret2.outputExpected = MakeTensor<T, 3>(outputTensorInfo2, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            11.0f, 12.0f,
            16.0f, 17.0f,
            21.0f, 22.0f,
            26.0f, 27.0f,

            41.0f, 42.0f,
            46.0f, 47.0f,
            51.0f, 52.0f,
            56.0f, 57.0f,

            71.0f, 72.0f,
            76.0f, 77.0f,
            81.0f, 82.0f,
            86.0f, 87.0f,
        })
    ));

    ret3.outputExpected = MakeTensor<T, 3>(outputTensorInfo3, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            3.0f, 4.0f, 5.0f,
            8.0f, 9.0f, 10.0f,
            13.0f, 14.0f, 15.0f,
            18.0f, 19.0f, 20.0f,
            23.0f, 24.0f, 25.0f,
            28.0f, 29.0f, 30.0f,

            33.0f, 34.0f, 35.0f,
            38.0f, 39.0f, 40.0f,
            43.0f, 44.0f, 45.0f,
            48.0f, 49.0f, 50.0f,
            53.0f, 54.0f, 55.0f,
            58.0f, 59.0f, 60.0f,
        })
    ));

    ret4.outputExpected = MakeTensor<T, 3>(outputTensorInfo4, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            63.0f, 64.0f, 65.0f,
            68.0f, 69.0f, 70.0f,
            73.0f, 74.0f, 75.0f,
            78.0f, 79.0f, 80.0f,
            83.0f, 84.0f, 85.0f,
            88.0f, 89.0f, 90.0f,
        })
    ));


    ret5.outputExpected = MakeTensor<T, 3>(outputTensorInfo5, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
            11.0f, 12.0f,
            16.0f, 17.0f,
            21.0f, 22.0f,
            26.0f, 27.0f,

            41.0f, 42.0f,
            46.0f, 47.0f,
            51.0f, 52.0f,
            56.0f, 57.0f,

            71.0f, 72.0f,
            76.0f, 77.0f,
            81.0f, 82.0f,
            86.0f, 87.0f,
        })
    ));

    std::vector<unsigned int> wOrigin1 = {0, 0, 0}; //extent of the window is defined by size of output[0]
    armnn::SplitterQueueDescriptor::ViewOrigin window1(wOrigin1);

    std::vector<unsigned int> wOrigin2 = {0, 2, 0}; //extent of the window is defined by size of output[1]
    armnn::SplitterQueueDescriptor::ViewOrigin window2(wOrigin2);

    std::vector<unsigned int> wOrigin3 = {0, 0, 2}; //extent of the window is defined by size of output[2]
    armnn::SplitterQueueDescriptor::ViewOrigin window3(wOrigin3);

    std::vector<unsigned int> wOrigin4 = {2, 0, 2}; //extent of the window is defined by size of output[3]
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
            workloadFactory.CreateSubTensorHandle(*inputHandle, outputTensorInfo3.GetShape(), wOrigin3.data()) :
            workloadFactory.CreateTensorHandle(outputTensorInfo3);

    std::unique_ptr<armnn::ITensorHandle> outputHandle4 =
        subTensorsSupported ?
            workloadFactory.CreateSubTensorHandle(*inputHandle, outputTensorInfo4.GetShape(), wOrigin4.data()) :
            workloadFactory.CreateTensorHandle(outputTensorInfo4);

    std::unique_ptr<armnn::ITensorHandle> outputHandle5 =
        subTensorsSupported ?
            workloadFactory.CreateSubTensorHandle(*inputHandle, outputTensorInfo5.GetShape(), wOrigin2.data()) :
            workloadFactory.CreateTensorHandle(outputTensorInfo5);

    armnn::SplitterQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo1, outputHandle1.get());
    AddOutputToWorkload(data, info, outputTensorInfo2, outputHandle2.get());
    AddOutputToWorkload(data, info, outputTensorInfo3, outputHandle3.get());
    AddOutputToWorkload(data, info, outputTensorInfo4, outputHandle4.get());
    AddOutputToWorkload(data, info, outputTensorInfo5, outputHandle5.get());

    data.m_ViewOrigins.push_back(window1);
    data.m_ViewOrigins.push_back(window2);
    data.m_ViewOrigins.push_back(window3);
    data.m_ViewOrigins.push_back(window4);
    //add window2 again (to have an overlapping split)
    data.m_ViewOrigins.push_back(window2);

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateSplitter(data, info);

    inputHandle->Allocate();
    outputHandle1->Allocate();
    outputHandle2->Allocate();
    outputHandle3->Allocate();
    outputHandle4->Allocate();
    outputHandle5->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&ret1.output[0][0][0], outputHandle1.get());
    CopyDataFromITensorHandle(&ret2.output[0][0][0], outputHandle2.get());
    CopyDataFromITensorHandle(&ret3.output[0][0][0], outputHandle3.get());
    CopyDataFromITensorHandle(&ret4.output[0][0][0], outputHandle4.get());
    CopyDataFromITensorHandle(&ret5.output[0][0][0], outputHandle5.get());

    std::vector<LayerTestResult<T,3>> ret = {ret1, ret2, ret3, ret4, ret5};

    return ret;
}


template <typename T>
LayerTestResult<T, 3> CopyViaSplitterTestImpl(armnn::IWorkloadFactory& workloadFactory, float qScale, int32_t qOffset)
{
    const armnn::TensorInfo tensorInfo({ 3, 6, 5 }, armnn::GetDataType<T>());
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
