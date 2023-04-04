//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <opaque/include/armnn_delegate.hpp>
#include <opaque/include/Version.hpp>

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

}

} // namespace armnnDelegate
