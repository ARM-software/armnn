//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DelegateOptionsTestHelper.hpp"

namespace armnnDelegate
{

TEST_SUITE("DelegateOptions")
{

TEST_CASE ("ArmnnDelegateOptimizerOptionsReduceFp32ToFp16")
{
    std::stringstream ss;
    {
        StreamRedirector redirect(std::cout, ss.rdbuf());

        std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
        std::vector<int32_t> tensorShape { 1, 2, 2, 1 };
        std::vector<float> inputData = { 1, 2, 3, 4 };
        std::vector<float> divData = { 2, 2, 3, 4 };
        std::vector<float> expectedResult = { 1, 2, 2, 2 };

        // Enable ReduceFp32ToFp16
        armnn::OptimizerOptions optimizerOptions(true, true, false, false);
        armnn::INetworkProperties networkProperties;
        armnnDelegate::DelegateOptions delegateOptions(backends, optimizerOptions, networkProperties);

        DelegateOptionTest<float>(::tflite::TensorType_FLOAT32,
                                  backends,
                                  tensorShape,
                                  inputData,
                                  inputData,
                                  divData,
                                  expectedResult,
                                  delegateOptions);
    }
    // ReduceFp32ToFp16 option is enabled
    CHECK(ss.str().find("convert_fp32_to_fp16") != std::string::npos);
    CHECK(ss.str().find("convert_fp16_to_fp32") != std::string::npos);
}

TEST_CASE ("ArmnnDelegateOptimizerOptionsDebug")
{
    std::stringstream ss;
    {
        StreamRedirector redirect(std::cout, ss.rdbuf());

        std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
        std::vector<int32_t> tensorShape { 1, 2, 2, 1 };
        std::vector<float> inputData = { 1, 2, 3, 4 };
        std::vector<float> divData = { 2, 2, 3, 4 };
        std::vector<float> expectedResult = { 1, 2, 2, 2 };

        // Enable Debug
        armnn::OptimizerOptions optimizerOptions(false, true, false, false);
        armnn::INetworkProperties networkProperties;
        armnnDelegate::DelegateOptions delegateOptions(backends, optimizerOptions, networkProperties);

        DelegateOptionTest<float>(::tflite::TensorType_FLOAT32,
                                  backends,
                                  tensorShape,
                                  inputData,
                                  inputData,
                                  divData,
                                  expectedResult,
                                  delegateOptions);
    }
    // Debug option triggered.
    CHECK(ss.str().find("layerGuid") != std::string::npos);
    CHECK(ss.str().find("layerName") != std::string::npos);
    CHECK(ss.str().find("outputSlot") != std::string::npos);
    CHECK(ss.str().find("shape") != std::string::npos);
    CHECK(ss.str().find("data") != std::string::npos);
}

TEST_CASE ("ArmnnDelegateOptimizerOptionsDebugFunction")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    std::vector<int32_t> tensorShape { 1, 2, 2, 1 };
    std::vector<float> inputData = { 1, 2, 3, 4 };
    std::vector<float> divData = { 2, 2, 3, 4 };
    std::vector<float> expectedResult = { 1, 2, 2, 2 };

    // Enable debug with debug callback function
    armnn::OptimizerOptions optimizerOptions(false, true, false, false);
    bool callback = false;
    auto mockCallback = [&](armnn::LayerGuid guid, unsigned int slotIndex, armnn::ITensorHandle* tensor)
    {
        armnn::IgnoreUnused(guid);
        armnn::IgnoreUnused(slotIndex);
        armnn::IgnoreUnused(tensor);
        callback = true;
    };

    armnn::INetworkProperties networkProperties;
    armnnDelegate::DelegateOptions delegateOptions(backends,
                                                   optimizerOptions,
                                                   networkProperties,
                                                   armnn::EmptyOptional(),
                                                   armnn::Optional<armnn::DebugCallbackFunction>(mockCallback));

    CHECK(!callback);

    DelegateOptionTest<float>(::tflite::TensorType_FLOAT32,
                              backends,
                              tensorShape,
                              inputData,
                              inputData,
                              divData,
                              expectedResult,
                              delegateOptions);

    // Check that the debug callback function was called.
    CHECK(callback);
}

TEST_CASE ("ArmnnDelegateOptimizerOptionsReduceFp32ToBf16")
{
    std::stringstream ss;
    {
        StreamRedirector redirect(std::cout, ss.rdbuf());

        ReduceFp32ToBf16TestImpl();
    }

    // ReduceFp32ToBf16 option is enabled
    CHECK(ss.str().find("convert_fp32_to_bf16") != std::string::npos);
}

TEST_CASE ("ArmnnDelegateOptimizerOptionsImport")
{
    std::vector<armnn::BackendId> backends = {  armnn::Compute::CpuAcc, armnn::Compute::CpuRef };
    std::vector<int32_t> tensorShape { 1, 2, 2, 1 };
    std::vector<uint8_t> inputData = { 1, 2, 3, 4 };
    std::vector<uint8_t> divData = { 2, 2, 3, 4 };
    std::vector<uint8_t> expectedResult = { 1, 2, 2, 2};

    armnn::OptimizerOptions optimizerOptions(false, false, false, true);
    armnn::INetworkProperties networkProperties(true, true);
    armnnDelegate::DelegateOptions delegateOptions(backends, optimizerOptions, networkProperties);

    DelegateOptionTest<uint8_t>(::tflite::TensorType_UINT8,
                                backends,
                                tensorShape,
                                inputData,
                                inputData,
                                divData,
                                expectedResult,
                                delegateOptions);
}

}

} // namespace armnnDelegate
