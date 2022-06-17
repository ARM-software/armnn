//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "ArmnnDriver"

#include <android-base/logging.h>
#include <android-base/scopeguard.h>
#include <dlfcn.h>

#include "NeuralNetworksShim.h"
#include "SupportLibrarySymbols.h"

#include <string>

using namespace std;

int main()
{
    /// The platform shim allows use of the armnn support library driver (arm-armnn-sl) to create a
    /// binderized vendor service (arm-armnn-shim) that is started at device startup

    NnApiSLDriverImpl* impl = ANeuralNetworks_getSLDriverImpl();
    if (impl == nullptr)
    {
        LOG(ERROR) << "ArmnnDriver: ANeuralNetworks_getSLDriverImpl returned nullptr!!!";
        return EXIT_FAILURE;
    }

    ANeuralNetworksShimDeviceInfo* deviceInfo;
    ANeuralNetworksShimDeviceInfo_create(&deviceInfo,
                                         /*deviceName=*/"arm-armnn-sl",
                                         /*serviceName=*/"arm-armnn-shim");
    const auto guardDeviceInfo = android::base::make_scope_guard(
            [deviceInfo] { ANeuralNetworksShimDeviceInfo_free(deviceInfo); });

    ANeuralNetworksShimRegistrationParams* params;
    ANeuralNetworksShimRegistrationParams_create(impl, &params);
    const auto guardParams = android::base::make_scope_guard(
            [params] { ANeuralNetworksShimRegistrationParams_free(params); });
    ANeuralNetworksShimRegistrationParams_addDeviceInfo(params, deviceInfo);
    ANeuralNetworksShimRegistrationParams_setNumberOfListenerThreads(params, 15);
    ANeuralNetworksShimRegistrationParams_registerAsLazyService(params, false);
    ANeuralNetworksShimRegistrationParams_fallbackToMinimumSupportDevice(params, false);

    auto result = ANeuralNetworksShim_registerSupportLibraryService(params);
    LOG(ERROR) << "ArmnnDriver: ANeuralNetworksShim_registerSupportLibraryService returned error status: " << result;

    return EXIT_FAILURE;
}
