//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <opaque/include/armnn_delegate.hpp>

namespace armnnOpaqueDelegate
{

TEST_SUITE("ArmnnOpaqueDelegate")
{

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
    auto opaqueDelegate = armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateCreate(&options);
    CHECK(opaqueDelegate);

    // Check Opaque Delegate can be deleted
    CHECK(opaqueDelegate->opaque_delegate_builder->data);
    armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateDelete(opaqueDelegate);
}

TEST_CASE ("DelegatePluginTest")
{
    // Use default settings until options have been enabled.
    flatbuffers::FlatBufferBuilder flatBufferBuilder;
    tflite::TFLiteSettingsBuilder tfliteSettingsBuilder(flatBufferBuilder);
    flatbuffers::Offset<tflite::TFLiteSettings> tfliteSettings = tfliteSettingsBuilder.Finish();
    flatBufferBuilder.Finish(tfliteSettings);
    const tflite::TFLiteSettings* settings = flatbuffers::GetRoot<tflite::TFLiteSettings>(
        flatBufferBuilder.GetBufferPointer());

    std::unique_ptr<tflite::delegates::DelegatePluginInterface> delegatePlugin =
        tflite::delegates::DelegatePluginRegistry::CreateByName("armnn_delegate", *settings);

    // Plugin is created correctly using armnn_delegate name.
    CHECK((delegatePlugin != nullptr));

    tflite::delegates::TfLiteDelegatePtr armnnDelegate = delegatePlugin->Create();

    // Armnn Opaque Delegate is created correctly.
    CHECK((armnnDelegate != nullptr));
    CHECK((armnnDelegate->opaque_delegate_builder != nullptr));
}

}
} // namespace armnnDelegate
