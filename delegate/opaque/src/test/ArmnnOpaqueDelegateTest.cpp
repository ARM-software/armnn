//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <opaque/include/armnn_delegate.hpp>

#include <tensorflow/lite/kernels/builtin_op_kernels.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include "tensorflow/lite/core/c/builtin_op_data.h"

namespace armnnOpaqueDelegate
{

TEST_SUITE("ArmnnOpaqueDelegate")
{

TEST_CASE ("ArmnnOpaqueDelegate_Registered")
{
    using namespace tflite;
    auto tfLiteInterpreter = std::make_unique<Interpreter>();

    tfLiteInterpreter->AddTensors(3);
    tfLiteInterpreter->SetInputs({0, 1});
    tfLiteInterpreter->SetOutputs({2});

    tfLiteInterpreter->SetTensorParametersReadWrite(0, kTfLiteFloat32, "input1", {1,2,2,1}, TfLiteQuantization());
    tfLiteInterpreter->SetTensorParametersReadWrite(1, kTfLiteFloat32, "input2", {1,2,2,1}, TfLiteQuantization());
    tfLiteInterpreter->SetTensorParametersReadWrite(2, kTfLiteFloat32, "output", {1,2,2,1}, TfLiteQuantization());

    TfLiteAddParams* addParams = reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
    addParams->activation = kTfLiteActNone;
    addParams->pot_scale_int16 = false;

    tflite::ops::builtin::BuiltinOpResolver opResolver;
    const TfLiteRegistration* opRegister = opResolver.FindOp(BuiltinOperator_ADD, 1);
    tfLiteInterpreter->AddNodeWithParameters({0, 1}, {2}, "", 0, addParams, opRegister);

    // Create the Armnn Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    std::vector<armnn::BackendOptions> backendOptions;
    backendOptions.emplace_back(
            armnn::BackendOptions{ "BackendName",
                                   {
                                           { "Option1", 42 },
                                           { "Option2", true }
                                   }}
    );

    armnnDelegate::DelegateOptions delegateOptions(backends, backendOptions);
    std::unique_ptr<TfLiteDelegate, decltype(&armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateDelete)>
            theArmnnDelegate(armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateCreate(delegateOptions),
                             armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateDelete);

    auto status = tfLiteInterpreter->ModifyGraphWithDelegate(std::move(theArmnnDelegate));
    CHECK(status == kTfLiteOk);
    CHECK(tfLiteInterpreter != nullptr);
}

TEST_CASE ("ArmnnOpaqueDelegate_OptimizerOptionsRegistered")
{
    using namespace tflite;
    auto tfLiteInterpreter = std::make_unique<Interpreter>();

    tfLiteInterpreter->AddTensors(3);
    tfLiteInterpreter->SetInputs({0, 1});
    tfLiteInterpreter->SetOutputs({2});

    tfLiteInterpreter->SetTensorParametersReadWrite(0, kTfLiteFloat32, "input1", {1,2,2,1}, TfLiteQuantization());
    tfLiteInterpreter->SetTensorParametersReadWrite(1, kTfLiteFloat32, "input2", {1,2,2,1}, TfLiteQuantization());
    tfLiteInterpreter->SetTensorParametersReadWrite(2, kTfLiteFloat32, "output", {1,2,2,1}, TfLiteQuantization());

    TfLiteAddParams* addParams = reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
    addParams->activation = kTfLiteActNone;
    addParams->pot_scale_int16 = false;

    tflite::ops::builtin::BuiltinOpResolver opResolver;
    const TfLiteRegistration* opRegister = opResolver.FindOp(BuiltinOperator_ADD, 1);
    tfLiteInterpreter->AddNodeWithParameters({0, 1}, {2}, "", 0, addParams, opRegister);

    // Create the Armnn Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };

    armnn::OptimizerOptionsOpaque optimizerOptions(true, true, false, true);

    armnnDelegate::DelegateOptions delegateOptions(backends, optimizerOptions);
    std::unique_ptr<TfLiteDelegate, decltype(&armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateDelete)>
                        theArmnnDelegate(armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateCreate(delegateOptions),
                                         armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateDelete);

    auto status = tfLiteInterpreter->ModifyGraphWithDelegate(std::move(theArmnnDelegate));
    CHECK(status == kTfLiteOk);
    CHECK(tfLiteInterpreter != nullptr);
}

TEST_CASE ("DelegateOptions_OpaqueDelegateDefault")
{
    // Check default options can be created
    auto options = armnnOpaqueDelegate::TfLiteArmnnDelegateOptionsDefault();
    armnnOpaqueDelegate::ArmnnOpaqueDelegate delegate(options);

    // Check version returns correctly
    auto version = delegate.GetVersion();
    CHECK_EQ(version, OPAQUE_DELEGATE_VERSION);

    auto* builder = delegate.GetDelegateBuilder();
    CHECK(builder);

    // Check Opaque delegate created
    auto opaqueDelegate = armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateCreate(options);
    CHECK(opaqueDelegate);

    // Check Opaque Delegate can be deleted
    CHECK(opaqueDelegate->opaque_delegate_builder->data);
    armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateDelete(opaqueDelegate);
}

TEST_CASE ("DelegatePluginTest")
{
    const char* backends = "CpuRef";
    bool fastmath = false;
    const char* additional_parameters = "allow-expanded-dims=true";

    flatbuffers::FlatBufferBuilder flatbuffer_builder;
    flatbuffers::Offset<tflite::ArmNNSettings>
            armnn_settings_offset = tflite::CreateArmNNSettingsDirect(flatbuffer_builder,
                                                                      backends,
                                                                      fastmath,
                                                                      additional_parameters);

    tflite::TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder);
    tflite_settings_builder.add_armnn_settings(armnn_settings_offset);
    flatbuffers::Offset<tflite::TFLiteSettings> tflite_settings_offset = tflite_settings_builder.Finish();
    flatbuffer_builder.Finish(tflite_settings_offset);

    const tflite::TFLiteSettings* tflite_settings = flatbuffers::GetRoot<tflite::TFLiteSettings>(
            flatbuffer_builder.GetBufferPointer());

    std::unique_ptr<tflite::delegates::DelegatePluginInterface> delegatePlugin =
        tflite::delegates::DelegatePluginRegistry::CreateByName("armnn_delegate", *tflite_settings);

    // Plugin is created correctly using armnn_delegate name.
    CHECK((delegatePlugin != nullptr));

    tflite::delegates::TfLiteDelegatePtr armnnDelegate = delegatePlugin->Create();

    // Armnn Opaque Delegate is created correctly.
    CHECK((armnnDelegate != nullptr));
    CHECK((armnnDelegate->opaque_delegate_builder != nullptr));
}

armnnDelegate::DelegateOptions BuildDelegateOptions(const char* backends,
                                                 bool fastmath,
                                                 const char* additional_parameters)
{
    flatbuffers::FlatBufferBuilder flatbuffer_builder;

    flatbuffers::Offset<tflite::ArmNNSettings>
        armnn_settings_offset = tflite::CreateArmNNSettingsDirect(flatbuffer_builder,
                                                                  backends,
                                                                  fastmath,
                                                                  additional_parameters);

    tflite::TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder);
    tflite_settings_builder.add_armnn_settings(armnn_settings_offset);
    flatbuffers::Offset<tflite::TFLiteSettings> tflite_settings_offset = tflite_settings_builder.Finish();
    flatbuffer_builder.Finish(tflite_settings_offset);

    const tflite::TFLiteSettings* tflite_settings = flatbuffers::GetRoot<tflite::TFLiteSettings>(
        flatbuffer_builder.GetBufferPointer());

    armnnDelegate::DelegateOptions delegateOptions = ParseArmNNSettings(tflite_settings);

    return delegateOptions;
}

unsigned int CountBackendOptions(armnn::BackendId backendId,
                                 armnnDelegate::DelegateOptions& delegateOptions,
                                 bool runtime = false)
{
    unsigned int count = 0;

    std::vector<armnn::BackendOptions> modelOptions = runtime ? delegateOptions.GetRuntimeOptions().m_BackendOptions
                                                              : delegateOptions.GetOptimizerOptions().GetModelOptions();
    for (const auto& backendOptions : modelOptions)
    {
        if (backendOptions.GetBackendId() == backendId)
        {
            count = backendOptions.GetOptionCount();
        }
    }

    return count;
}

bool GetBackendOption(armnn::BackendId backendId,
                      armnnDelegate::DelegateOptions& delegateOptions,
                      std::string& optionName,
                      armnn::BackendOptions::BackendOption& backendOption,
                      bool runtime = false)
{
    bool result = false;

    std::vector<armnn::BackendOptions> modelOptions = runtime ? delegateOptions.GetRuntimeOptions().m_BackendOptions
                                                              : delegateOptions.GetOptimizerOptions().GetModelOptions();

    for (const auto& backendOptions : modelOptions)
    {
        if (backendOptions.GetBackendId() == backendId)
        {
            for (size_t i = 0; i < backendOptions.GetOptionCount(); ++i)
            {
                const armnn::BackendOptions::BackendOption& option = backendOptions.GetOption(i);
                if (option.GetName() == optionName)
                {
                    backendOption = option;
                    result = true;
                    break;
                }
            }
        }
    }

    return result;
}

TEST_CASE ("ParseArmNNSettings_backend")
{
    {
        armnnDelegate::DelegateOptions delegateOptions = BuildDelegateOptions("CpuRef,GpuAcc", false, nullptr);

        std::vector<armnn::BackendId> expectedBackends = {"CpuRef", "GpuAcc"};
        CHECK_EQ(expectedBackends, delegateOptions.GetBackends());
    }
    {
        armnnDelegate::DelegateOptions delegateOptions = BuildDelegateOptions("GpuAcc", false, nullptr);

        std::vector<armnn::BackendId> expectedBackends = {"GpuAcc"};
        CHECK_EQ(expectedBackends, delegateOptions.GetBackends());
    }
}

TEST_CASE ("ParseArmNNSettings_fastmath")
{
    // Test fastmath true in both backends
    {
        armnnDelegate::DelegateOptions delegateOptions = BuildDelegateOptions("CpuAcc,GpuAcc", true, nullptr);

        std::vector<armnn::BackendId> expectedBackends = {"CpuAcc", "GpuAcc"};
        CHECK_EQ(expectedBackends, delegateOptions.GetBackends());
        CHECK_EQ(CountBackendOptions(armnn::Compute::CpuAcc, delegateOptions), 1);
        CHECK_EQ(CountBackendOptions(armnn::Compute::CpuAcc, delegateOptions), 1);

        armnn::BackendOptions::BackendOption backendOption("", false);
        std::string optionName = "FastMathEnabled";
        CHECK_EQ(GetBackendOption(armnn::Compute::CpuAcc, delegateOptions, optionName, backendOption), true);
        CHECK_EQ(backendOption.GetValue().AsBool(), true);
        CHECK_EQ(backendOption.GetName(), optionName);
        CHECK_EQ(GetBackendOption(armnn::Compute::GpuAcc, delegateOptions, optionName, backendOption), true);
        CHECK_EQ(backendOption.GetValue().AsBool(), true);
        CHECK_EQ(backendOption.GetName(), optionName);
    }

    // Test fastmath true in one backend
    {
        armnnDelegate::DelegateOptions delegateOptions = BuildDelegateOptions("CpuAcc,CpuRef", true, nullptr);

        std::vector<armnn::BackendId> expectedBackends = {"CpuAcc", "CpuRef"};
        CHECK_EQ(expectedBackends, delegateOptions.GetBackends());
        CHECK_EQ(CountBackendOptions(armnn::Compute::CpuAcc, delegateOptions), 1);
        CHECK_EQ(CountBackendOptions(armnn::Compute::CpuRef, delegateOptions), 0);

        armnn::BackendOptions::BackendOption backendOption("", false);
        std::string optionName = "FastMathEnabled";
        CHECK_EQ(GetBackendOption(armnn::Compute::CpuAcc, delegateOptions, optionName, backendOption), true);
        CHECK_EQ(backendOption.GetValue().AsBool(), true);
        CHECK_EQ(backendOption.GetName(), optionName);
        CHECK_EQ(GetBackendOption(armnn::Compute::CpuRef, delegateOptions, optionName, backendOption), false);
    }

    // Test fastmath false
    {
        armnnDelegate::DelegateOptions delegateOptions = BuildDelegateOptions("GpuAcc", false, nullptr);

        std::vector<armnn::BackendId> expectedBackends = {"GpuAcc"};
        CHECK_EQ(expectedBackends, delegateOptions.GetBackends());
        CHECK_EQ(CountBackendOptions(armnn::Compute::GpuAcc, delegateOptions), 1);

        armnn::BackendOptions::BackendOption backendOption("", false);
        std::string optionName = "FastMathEnabled";
        CHECK_EQ(GetBackendOption(armnn::Compute::GpuAcc, delegateOptions, optionName, backendOption), true);
        CHECK_EQ(backendOption.GetValue().AsBool(), false);
        CHECK_EQ(backendOption.GetName(), optionName);
    }
}

TEST_CASE ("ParseArmNNSettings_additional_options_raw")
{
    const char* backends = "GpuAcc";
    bool fastmath = false;
    const char* additional_parameters = "allow-expanded-dims=true";

    flatbuffers::FlatBufferBuilder flatbuffer_builder;
    flatbuffers::Offset<tflite::ArmNNSettings>
        armnn_settings_offset = tflite::CreateArmNNSettingsDirect(flatbuffer_builder,
                                                                  backends,
                                                                  fastmath,
                                                                  additional_parameters);

    tflite::TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder);
    tflite_settings_builder.add_armnn_settings(armnn_settings_offset);
    flatbuffers::Offset<tflite::TFLiteSettings> tflite_settings_offset = tflite_settings_builder.Finish();
    flatbuffer_builder.Finish(tflite_settings_offset);

    const tflite::TFLiteSettings* tflite_settings = flatbuffers::GetRoot<tflite::TFLiteSettings>(
        flatbuffer_builder.GetBufferPointer());
    CHECK((tflite_settings->armnn_settings()->additional_parameters() != nullptr));
    CHECK_EQ(tflite_settings->armnn_settings()->additional_parameters()->str(), additional_parameters);

    armnnDelegate::DelegateOptions delegateOptions = ParseArmNNSettings(tflite_settings);
    CHECK_EQ(delegateOptions.GetOptimizerOptions().GetAllowExpandedDims(), true);
}

TEST_CASE ("ParseArmNNSettings_additional_options")
{
    std::string options = "number-of-threads=29,"              // optimizer-backend option only Cpu
                          "gpu-kernel-profiling-enabled=true," // runtime-backend option only GPU
                          "allow-expanded-dims=true,"          // optimizer option
                          "logging-severity=debug,"            // option
                          "counter-capture-period=100u";       // runtime-profiling option
    armnnDelegate::DelegateOptions delegateOptions = BuildDelegateOptions("CpuAcc,GpuAcc", false, options.c_str());

    // Variables to be used in all checks
    armnn::BackendOptions::BackendOption backendOption("", false);
    std::string optionName;

    // number-of-threads
    CHECK_EQ(CountBackendOptions(armnn::Compute::CpuAcc, delegateOptions), 1);
    optionName = "NumberOfThreads";
    CHECK_EQ(GetBackendOption(armnn::Compute::CpuAcc, delegateOptions, optionName, backendOption), true);
    CHECK_EQ(backendOption.GetValue().AsUnsignedInt(), 29);
    CHECK_EQ(backendOption.GetName(), optionName);

    // gpu-kernel-profiling-enabled
    CHECK_EQ(CountBackendOptions(armnn::Compute::GpuAcc, delegateOptions, true), 1);
    optionName = "KernelProfilingEnabled";
    CHECK_EQ(GetBackendOption(armnn::Compute::GpuAcc, delegateOptions, optionName, backendOption, true), true);
    CHECK_EQ(backendOption.GetValue().AsBool(), true);
    CHECK_EQ(backendOption.GetName(), optionName);

    // allow-expanded-dims
    CHECK_EQ(delegateOptions.GetOptimizerOptions().GetAllowExpandedDims(), true);

    // logging-severity
    CHECK_EQ(delegateOptions.GetLoggingSeverity(), armnn::LogSeverity::Debug);

    // counter-capture-period
    CHECK_EQ(delegateOptions.GetRuntimeOptions().m_ProfilingOptions.m_CapturePeriod, 100);
}

TEST_CASE ("ParseArmNNSettings_additional_options_regex")
{
    std::string options = "allow-expanded-dims= true, "              // optimizer option
                          "number-of-threads =29 ,"                  // optimizer-backend option only Cpu
                          "logging-severity   =   trace   ,   "      // option
                          "counter-capture-period       =     100u"; // runtime-profiling option
    armnnDelegate::DelegateOptions delegateOptions = BuildDelegateOptions("GpuAcc", false, options.c_str());

    // Variables to be used in all checks
    armnn::BackendOptions::BackendOption backendOption("", false);
    std::string optionName;

    std::vector<armnn::BackendId> expectedBackends = {"GpuAcc"};
    CHECK_EQ(expectedBackends, delegateOptions.GetBackends());

    // enable-fast-math
    CHECK_EQ(CountBackendOptions(armnn::Compute::GpuAcc, delegateOptions), 1);
    optionName = "FastMathEnabled";
    CHECK_EQ(GetBackendOption(armnn::Compute::CpuRef, delegateOptions, optionName, backendOption), false);
    CHECK_EQ(GetBackendOption(armnn::Compute::CpuAcc, delegateOptions, optionName, backendOption), false);
    CHECK_EQ(GetBackendOption(armnn::Compute::GpuAcc, delegateOptions, optionName, backendOption), true);
    CHECK_EQ(backendOption.GetValue().AsBool(), false);
    CHECK_EQ(backendOption.GetName(), optionName);

    // allow-expanded-dims
    CHECK_EQ(delegateOptions.GetOptimizerOptions().GetAllowExpandedDims(), true);

    // number-of-threads not saved anywhere, as it is a parameter only valid for CpuAcc
    optionName="number-of-threads";
    CHECK_EQ(GetBackendOption(armnn::Compute::CpuAcc, delegateOptions, optionName, backendOption), false);
    CHECK_EQ(GetBackendOption(armnn::Compute::GpuAcc, delegateOptions, optionName, backendOption), false);

    // logging-severity
    CHECK_EQ(delegateOptions.GetLoggingSeverity(), armnn::LogSeverity::Trace);

    // counter-capture-period
    CHECK_EQ(delegateOptions.GetRuntimeOptions().m_ProfilingOptions.m_CapturePeriod, 100);
}

TEST_CASE ("ParseArmNNSettings_additional_options_incorrect")
{
    std::string options = "number-of-thread=29"; // The correct one would be "number-of-threads" in plural

    CHECK_THROWS(BuildDelegateOptions("CpuAcc,GpuAcc", false, options.c_str()));
}

}
} // namespace armnnDelegate
