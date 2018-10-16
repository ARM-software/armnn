//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <backends/test/LayerTests.hpp>
#include <backends/test/TensorCopyUtils.hpp>
#include <backends/test/WorkloadTestUtils.hpp>

#include <armnn/test/TensorHelpers.hpp>

#include <boost/multi_array.hpp>

namespace
{

LayerTestResult<float, 4> MemCopyTest(armnn::IWorkloadFactory& srcWorkloadFactory,
                                      armnn::IWorkloadFactory& dstWorkloadFactory,
                                      bool withSubtensors)
{
    const std::array<unsigned int, 4> shapeData = { { 1u, 1u, 6u, 5u } };
    const armnn::TensorShape tensorShape(4, shapeData.data());
    const armnn::TensorInfo tensorInfo(tensorShape, armnn::DataType::Float32);
    boost::multi_array<float, 4> inputData = MakeTensor<float, 4>(tensorInfo, std::vector<float>(
        {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,

            6.0f, 7.0f, 8.0f, 9.0f, 10.0f,

            11.0f, 12.0f, 13.0f, 14.0f, 15.0f,

            16.0f, 17.0f, 18.0f, 19.0f, 20.0f,

            21.0f, 22.0f, 23.0f, 24.0f, 25.0f,

            26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
        })
    );

    LayerTestResult<float, 4> ret(tensorInfo);
    ret.outputExpected = inputData;

    boost::multi_array<float, 4> outputData(shapeData);

    auto inputTensorHandle = srcWorkloadFactory.CreateTensorHandle(tensorInfo);
    auto outputTensorHandle = dstWorkloadFactory.CreateTensorHandle(tensorInfo);

    AllocateAndCopyDataToITensorHandle(inputTensorHandle.get(), inputData.data());
    outputTensorHandle->Allocate();

    armnn::MemCopyQueueDescriptor memCopyQueueDesc;
    armnn::WorkloadInfo workloadInfo;

    const unsigned int origin[4] = {};

    auto workloadInput = (withSubtensors && srcWorkloadFactory.SupportsSubTensors())
                         ? srcWorkloadFactory.CreateSubTensorHandle(*inputTensorHandle, tensorShape, origin)
                         : std::move(inputTensorHandle);
    auto workloadOutput = (withSubtensors && dstWorkloadFactory.SupportsSubTensors())
                          ? dstWorkloadFactory.CreateSubTensorHandle(*outputTensorHandle, tensorShape, origin)
                          : std::move(outputTensorHandle);

    AddInputToWorkload(memCopyQueueDesc, workloadInfo, tensorInfo, workloadInput.get());
    AddOutputToWorkload(memCopyQueueDesc, workloadInfo, tensorInfo, workloadOutput.get());

    dstWorkloadFactory.CreateMemCopy(memCopyQueueDesc, workloadInfo)->Execute();

    CopyDataFromITensorHandle(outputData.data(), workloadOutput.get());
    ret.output = outputData;

    return ret;
}

template<typename SrcWorkloadFactory, typename DstWorkloadFactory>
LayerTestResult<float, 4> MemCopyTest(bool withSubtensors)
{
    SrcWorkloadFactory srcWorkloadFactory;
    DstWorkloadFactory dstWorkloadFactory;

    return MemCopyTest(srcWorkloadFactory, dstWorkloadFactory, withSubtensors);
}

} // anonymous namespace
