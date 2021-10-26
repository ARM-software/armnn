//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DelegateOptionsTestHelper.hpp"
#include <common/include/ProfilingGuid.hpp>
#include <armnnUtils/Filesystem.hpp>

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
        armnnDelegate::DelegateOptions delegateOptions(backends, optimizerOptions);

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
        armnnDelegate::DelegateOptions delegateOptions(backends, optimizerOptions);

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

    armnn::INetworkProperties networkProperties(false, armnn::MemorySource::Undefined, armnn::MemorySource::Undefined);
    armnnDelegate::DelegateOptions delegateOptions(backends,
                                                   optimizerOptions,
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
    std::vector<uint8_t> expectedResult = { 1, 2, 2, 2 };

    armnn::OptimizerOptions optimizerOptions(false, false, false, true);
    armnnDelegate::DelegateOptions delegateOptions(backends, optimizerOptions);

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

TEST_SUITE("DelegateOptions_CpuAccTests")
{

TEST_CASE ("ArmnnDelegateModelOptions_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    std::vector<int32_t> tensorShape { 1, 2, 2, 1 };
    std::vector<float> inputData = { 1, 2, 3, 4 };
    std::vector<float> divData = { 2, 2, 3, 4 };
    std::vector<float> expectedResult = { 1, 2, 2, 2 };

    unsigned int numberOfThreads = 2;

    armnn::ModelOptions modelOptions;
    armnn::BackendOptions cpuAcc("CpuAcc",
                                 {
                                         { "FastMathEnabled", true },
                                         { "NumberOfThreads", numberOfThreads }
                                 });
    modelOptions.push_back(cpuAcc);

    armnn::OptimizerOptions optimizerOptions(false, false, false, false, modelOptions);
    armnnDelegate::DelegateOptions delegateOptions(backends, optimizerOptions);

    DelegateOptionTest<float>(::tflite::TensorType_FLOAT32,
                              backends,
                              tensorShape,
                              inputData,
                              inputData,
                              divData,
                              expectedResult,
                              delegateOptions);
}

TEST_CASE ("ArmnnDelegateSerializeToDot")
{
    const fs::path filename(fs::temp_directory_path() / "ArmnnDelegateSerializeToDot.dot");
    if ( fs::exists(filename) )
    {
        fs::remove(filename);
    }
    std::stringstream ss;
    {
        StreamRedirector redirect(std::cout, ss.rdbuf());

        std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
        std::vector<int32_t> tensorShape { 1, 2, 2, 1 };
        std::vector<float> inputData = { 1, 2, 3, 4 };
        std::vector<float> divData = { 2, 2, 3, 4 };
        std::vector<float> expectedResult = { 1, 2, 2, 2 };

        armnn::OptimizerOptions optimizerOptions(false, false, false, false);
        armnnDelegate::DelegateOptions delegateOptions(backends, optimizerOptions);
        // Enable serialize to dot by specifying the target file name.
        delegateOptions.SetSerializeToDot(filename);
        DelegateOptionTest<float>(::tflite::TensorType_FLOAT32,
                                  backends,
                                  tensorShape,
                                  inputData,
                                  inputData,
                                  divData,
                                  expectedResult,
                                  delegateOptions);
    }
    CHECK(fs::exists(filename));
    // The file should have a size greater than 0 bytes.
    CHECK(fs::file_size(filename) > 0);
    // Clean up.
    fs::remove(filename);
}

void CreateFp16StringParsingTestRun(std::vector<std::string>& keys,
                                    std::vector<std::string>& values,
                                    std::stringstream& ss)
{
    StreamRedirector redirect(std::cout, ss.rdbuf());

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    std::vector<int32_t> tensorShape { 1, 2, 2, 1 };
    std::vector<float> inputData = { 1, 2, 3, 4 };
    std::vector<float> divData = { 2, 2, 3, 4 };
    std::vector<float> expectedResult = { 1, 2, 2, 2 };

    // Create options_keys and options_values char array
    size_t num_options = keys.size();
    std::unique_ptr<const char*> options_keys =
            std::unique_ptr<const char*>(new const char*[num_options + 1]);
    std::unique_ptr<const char*> options_values =
            std::unique_ptr<const char*>(new const char*[num_options + 1]);
    for (size_t i=0; i<num_options; ++i)
    {
        options_keys.get()[i]   = keys[i].c_str();
        options_values.get()[i] = values[i].c_str();
    }

    armnnDelegate::DelegateOptions delegateOptions(options_keys.get(), options_values.get(), num_options, nullptr);
    DelegateOptionTest<float>(::tflite::TensorType_FLOAT32,
                              backends,
                              tensorShape,
                              inputData,
                              inputData,
                              divData,
                              expectedResult,
                              delegateOptions);
}

TEST_CASE ("ArmnnDelegateStringParsingOptionReduceFp32ToFp16")
{
    SUBCASE("Fp16=1")
    {
        std::stringstream ss;
        std::vector<std::string> keys   {  "backends", "debug-data", "reduce-fp32-to-fp16", "logging-severity"};
        std::vector<std::string> values {    "CpuRef",          "1",                   "1",             "info"};
        CreateFp16StringParsingTestRun(keys, values, ss);
        CHECK(ss.str().find("convert_fp32_to_fp16") != std::string::npos);
        CHECK(ss.str().find("convert_fp16_to_fp32") != std::string::npos);
    }
    SUBCASE("Fp16=true")
    {
        std::stringstream ss;
        std::vector<std::string> keys   {  "backends", "debug-data", "reduce-fp32-to-fp16"};
        std::vector<std::string> values {    "CpuRef",       "TRUE",                "true"};
        CreateFp16StringParsingTestRun(keys, values, ss);
        CHECK(ss.str().find("convert_fp32_to_fp16") != std::string::npos);
        CHECK(ss.str().find("convert_fp16_to_fp32") != std::string::npos);
    }
    SUBCASE("Fp16=True")
    {
        std::stringstream ss;
        std::vector<std::string> keys   {  "backends", "debug-data", "reduce-fp32-to-fp16"};
        std::vector<std::string> values {    "CpuRef",       "true",                "True"};
        CreateFp16StringParsingTestRun(keys, values, ss);
        CHECK(ss.str().find("convert_fp32_to_fp16") != std::string::npos);
        CHECK(ss.str().find("convert_fp16_to_fp32") != std::string::npos);
    }
    SUBCASE("Fp16=0")
    {
        std::stringstream ss;
        std::vector<std::string> keys   {  "backends", "debug-data", "reduce-fp32-to-fp16"};
        std::vector<std::string> values {    "CpuRef",       "true",                   "0"};
        CreateFp16StringParsingTestRun(keys, values, ss);
        CHECK(ss.str().find("convert_fp32_to_fp16") == std::string::npos);
        CHECK(ss.str().find("convert_fp16_to_fp32") == std::string::npos);
    }
    SUBCASE("Fp16=false")
    {
        std::stringstream ss;
        std::vector<std::string> keys   {  "backends", "debug-data", "reduce-fp32-to-fp16"};
        std::vector<std::string> values {    "CpuRef",     "1",               "false"};
        CreateFp16StringParsingTestRun(keys, values, ss);
        CHECK(ss.str().find("convert_fp32_to_fp16") == std::string::npos);
        CHECK(ss.str().find("convert_fp16_to_fp32") == std::string::npos);
    }
}


}

} // namespace armnnDelegate
