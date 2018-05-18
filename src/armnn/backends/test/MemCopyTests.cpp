//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include <boost/test/unit_test.hpp>
#include <boost/multi_array.hpp>

#include "armnn/ArmNN.hpp"
#include "backends/RefWorkloadFactory.hpp"
#if ARMCOMPUTECL_ENABLED
#include "backends/ClWorkloadFactory.hpp"
#endif
#if ARMCOMPUTENEON_ENABLED
#include "backends/NeonWorkloadFactory.hpp"
#endif
#include "backends/CpuTensorHandle.hpp"
#include "test/TensorHelpers.hpp"

#include "TensorCopyUtils.hpp"
#include "WorkloadTestUtils.hpp"

BOOST_AUTO_TEST_SUITE(MemCopyTestSuite)

void MemCopyTest(armnn::IWorkloadFactory& srcWorkloadFactory, armnn::IWorkloadFactory& dstWorkloadFactory,
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

    BOOST_TEST(CompareTensors(inputData, outputData));
}

template <typename SrcWorkloadFactory, typename DstWorkloadFactory>
void MemCopyTest(bool withSubtensors)
{
    SrcWorkloadFactory srcWorkloadFactory;
    DstWorkloadFactory dstWorkloadFactory;
    MemCopyTest(srcWorkloadFactory, dstWorkloadFactory, withSubtensors);
}

#if ARMCOMPUTECL_ENABLED

BOOST_AUTO_TEST_CASE(CopyBetweenCpuAndGpu)
{
    MemCopyTest<armnn::RefWorkloadFactory, armnn::ClWorkloadFactory>(false);
}

BOOST_AUTO_TEST_CASE(CopyBetweenGpuAndCpu)
{
    MemCopyTest<armnn::ClWorkloadFactory, armnn::RefWorkloadFactory>(false);
}

BOOST_AUTO_TEST_CASE(CopyBetweenCpuAndGpuWithSubtensors)
{
    MemCopyTest<armnn::RefWorkloadFactory, armnn::ClWorkloadFactory>(true);
}

BOOST_AUTO_TEST_CASE(CopyBetweenGpuAndCpuWithSubtensors)
{
    MemCopyTest<armnn::ClWorkloadFactory, armnn::RefWorkloadFactory>(true);
}

#endif // ARMCOMPUTECL_ENABLED

#if ARMCOMPUTENEON_ENABLED

BOOST_AUTO_TEST_CASE(CopyBetweenCpuAndNeon)
{
    MemCopyTest<armnn::RefWorkloadFactory, armnn::NeonWorkloadFactory>(false);
}

BOOST_AUTO_TEST_CASE(CopyBetweenNeonAndCpu)
{
    MemCopyTest<armnn::NeonWorkloadFactory, armnn::RefWorkloadFactory>(false);
}

BOOST_AUTO_TEST_CASE(CopyBetweenCpuAndNeonWithSubtensors)
{
    MemCopyTest<armnn::RefWorkloadFactory, armnn::NeonWorkloadFactory>(true);
}

BOOST_AUTO_TEST_CASE(CopyBetweenNeonAndCpuWithSubtensors)
{
    MemCopyTest<armnn::NeonWorkloadFactory, armnn::RefWorkloadFactory>(true);
}

#endif // ARMCOMPUTENEON_ENABLED

#if ARMCOMPUTECL_ENABLED && ARMCOMPUTENEON_ENABLED

BOOST_AUTO_TEST_CASE(CopyBetweenNeonAndGpu)
{
    MemCopyTest<armnn::NeonWorkloadFactory, armnn::ClWorkloadFactory>(false);
}

BOOST_AUTO_TEST_CASE(CopyBetweenGpuAndNeon)
{
    MemCopyTest<armnn::ClWorkloadFactory, armnn::NeonWorkloadFactory>(false);
}

BOOST_AUTO_TEST_CASE(CopyBetweenNeonAndGpuWithSubtensors)
{
    MemCopyTest<armnn::NeonWorkloadFactory, armnn::ClWorkloadFactory>(true);
}

BOOST_AUTO_TEST_CASE(CopyBetweenGpuAndNeonWithSubtensors)
{
    MemCopyTest<armnn::ClWorkloadFactory, armnn::NeonWorkloadFactory>(true);
}

#endif

BOOST_AUTO_TEST_SUITE_END()
