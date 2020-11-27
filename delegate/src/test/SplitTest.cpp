//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SplitTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

// SPLIT Operator
void SplitUint8Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> axisShape { 1 };
    std::vector<int32_t> inputShape { 2, 2, 2, 2} ;
    std::vector<int32_t> outputShape0 { 2, 2, 2, 1 };
    std::vector<int32_t> outputShape1 { 2, 2, 2, 1 };
    std::vector<std::vector<int32_t>> outputShapes{ outputShape0, outputShape1 };

    std::vector<int32_t> axisData { 3 };  // Axis
    std::vector<uint8_t> inputValues { 1, 2, 3, 4, 5, 6, 7, 8,
                                       9, 10, 11, 12, 13, 14, 15, 16 }; // Input


    std::vector<uint8_t> expectedOutputValues0 { 1, 3, 5, 7, 9, 11, 13, 15 };
    std::vector<uint8_t> expectedOutputValues1 { 2, 4, 6, 8, 10, 12, 14, 16 };
    std::vector<std::vector<uint8_t>> expectedOutputValues{ expectedOutputValues0, expectedOutputValues1 };

    int32_t numSplits = 2;

    SplitTest<uint8_t>(::tflite::TensorType_UINT8,
                       backends,
                       axisShape,
                       inputShape,
                       outputShapes,
                       axisData,
                       inputValues,
                       expectedOutputValues,
                       numSplits);
}

void SplitFp32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> axisShape { 1 };
    std::vector<int32_t> inputShape { 2, 2, 2, 2 };
    std::vector<int32_t> outputShape0 { 2, 1, 2, 2 };
    std::vector<int32_t> outputShape1 { 2, 1, 2, 2 };
    std::vector<std::vector<int32_t>> outputShapes{ outputShape0, outputShape1 };

    std::vector<int32_t> axisData { 1 };  // Axis
    std::vector<float> inputValues { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                     9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f  }; // Input


    std::vector<float> expectedOutputValues0 { 1.0f, 2.0f, 3.0f, 4.0f, 9.0f, 10.0f, 11.0f, 12.0f };
    std::vector<float> expectedOutputValues1 { 5.0f, 6.0f, 7.0f, 8.0f, 13.0f, 14.0f, 15.0f, 16.0f };
    std::vector<std::vector<float>> expectedOutputValues{ expectedOutputValues0, expectedOutputValues1 };

    int32_t numSplits = 2;

    SplitTest<float>(::tflite::TensorType_FLOAT32,
                     backends,
                     axisShape,
                     inputShape,
                     outputShapes,
                     axisData,
                     inputValues,
                     expectedOutputValues,
                     numSplits);
}

// SPLIT Test Suite
TEST_SUITE("SPLIT_CpuRefTests")
{

TEST_CASE ("SPLIT_Uint8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    SplitUint8Test(backends);
}

TEST_CASE ("SPLIT_Fp32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    SplitFp32Test(backends);
}

}

TEST_SUITE("SPLIT_CpuAccTests")
{

TEST_CASE ("SPLIT_Uint8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    SplitUint8Test(backends);
}

TEST_CASE ("SPLIT_Fp32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    SplitFp32Test(backends);
}

}

TEST_SUITE("SPLIT_GpuAccTests")
{

TEST_CASE ("SPLIT_Uint8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    SplitUint8Test(backends);
}

TEST_CASE ("SPLIT_Fp32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    SplitFp32Test(backends);
}

}
// End of SPLIT Test Suite

// SPLIT_V Operator
void SplitVUint8Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> axisShape { 1 };
    std::vector<int32_t> inputShape { 2, 4, 2, 2 };
    std::vector<int32_t> splitsShape { 2 };
    std::vector<int32_t> outputShape0 { 2, 3, 2, 2 };
    std::vector<int32_t> outputShape1 { 2, 1, 2, 2 };
    std::vector<std::vector<int32_t>> outputShapes{ outputShape0, outputShape1 };

    std::vector<int32_t> axisData { 1 };    // Axis
    std::vector<int32_t> splitsData { 3, 1 };  // Splits
    std::vector<uint8_t> inputValues { 1, 2, 3, 4, 5, 6, 7, 8,
                                     9, 10, 11, 12, 13, 14, 15, 16,
                                     17, 18, 19, 20, 21, 22, 23, 24,
                                     25, 26, 27, 28, 29, 30, 31, 32   }; // Input


    std::vector<uint8_t> expectedOutputValues0 { 1, 2, 3, 4, 5, 6, 7, 8,
                                               9, 10, 11, 12, 17, 18, 19, 20,
                                               21, 22, 23, 24, 25, 26, 27, 28 };
    std::vector<uint8_t> expectedOutputValues1 { 13, 14, 15, 16, 29, 30, 31, 32 };
    std::vector<std::vector<uint8_t>> expectedOutputValues{ expectedOutputValues0, expectedOutputValues1 };

    int32_t numSplits = 2;

    SplitVTest<uint8_t>(::tflite::TensorType_UINT8,
                        backends,
                        inputShape,
                        splitsShape,
                        axisShape,
                        outputShapes,
                        inputValues,
                        splitsData,
                        axisData,
                        expectedOutputValues,
                        numSplits);
}

void SplitVFp32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> axisShape { 1 };
    std::vector<int32_t> inputShape { 2, 4, 2, 2 };
    std::vector<int32_t> splitsShape { 2 };
    std::vector<int32_t> outputShape0 { 2, 3, 2, 2 };
    std::vector<int32_t> outputShape1 { 2, 1, 2, 2 };
    std::vector<std::vector<int32_t>> outputShapes{ outputShape0, outputShape1 };

    std::vector<int32_t> axisData { 1 };    // Axis
    std::vector<int32_t> splitsData { 3, 1 };  // Splits
    std::vector<float> inputValues { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                     9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                                     17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                                     25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f   }; // Input


    std::vector<float> expectedOutputValues0 { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                               9.0f, 10.0f, 11.0f, 12.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                                               21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f };
    std::vector<float> expectedOutputValues1 { 13.0f, 14.0f, 15.0f, 16.0f, 29.0f, 30.0f, 31.0f, 32.0f };
    std::vector<std::vector<float>> expectedOutputValues{ expectedOutputValues0, expectedOutputValues1 };

    int32_t numSplits = 2;

    SplitVTest<float>(::tflite::TensorType_FLOAT32,
                      backends,
                      inputShape,
                      splitsShape,
                      axisShape,
                      outputShapes,
                      inputValues,
                      splitsData,
                      axisData,
                      expectedOutputValues,
                      numSplits);
}

// SPLIT_V Test Suite
TEST_SUITE("SPLIT_V_CpuRefTests")
{

TEST_CASE ("SPLIT_V_Uint8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    SplitVUint8Test(backends);
}

TEST_CASE ("SPLIT_V_Fp32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    SplitVFp32Test(backends);
}

}

TEST_SUITE("SPLIT_V_CpuAccTests")
{

TEST_CASE ("SPLIT_V_Uint8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    SplitVUint8Test(backends);
}

TEST_CASE ("SPLIT_V_Fp32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    SplitVFp32Test(backends);
}

}

TEST_SUITE("SPLIT_V_GpuAccTests")
{

TEST_CASE ("SPLIT_V_Uint8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    SplitVUint8Test(backends);
}

TEST_CASE ("SPLIT_V_Fp32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    SplitVFp32Test(backends);
}

}
// End of SPLIT_V Test Suite

} // namespace armnnDelegate