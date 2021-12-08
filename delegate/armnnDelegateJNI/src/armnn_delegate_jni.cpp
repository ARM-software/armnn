//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn_delegate.hpp>
#include <DelegateOptions.hpp>

#if defined(ARMCOMPUTECL_ENABLED)
#include <arm_compute/core/CL/OpenCL.h>
#endif

#include <jni.h>
#include <string>

extern "C" {

/// Creates an Arm NN Delegate object.
/// Options are passed in form of String arrays. For details about what options_keys and option_values
/// are supported please see:
//  armnnDelegate::DelegateOptions::DelegateOptions(char const* const*, char const* const*,size_t,void (*)(const char*))
JNIEXPORT jlong
JNICALL Java_com_arm_armnn_delegate_ArmnnDelegate_createDelegate(JNIEnv* env,
                                                                 jclass clazz,
                                                                 jobjectArray optionKeys,
                                                                 jobjectArray optionValues)
{
    int numOptions = env->GetArrayLength(optionKeys);
    const char* nativeOptionKeys[numOptions];
    const char* nativeOptionValues[numOptions];

    jstring jKeyStrings[numOptions];
    jstring jValueStrings[numOptions];

    // Convert java array of string into char so we can make use of it in cpp code
    for (int i = 0; i < numOptions; i++)
    {
        jKeyStrings[i] = static_cast<jstring>(env->GetObjectArrayElement(optionKeys, i));
        jValueStrings[i] = static_cast<jstring>(env->GetObjectArrayElement(optionValues, i));

        nativeOptionKeys[i] = env->GetStringUTFChars(jKeyStrings[i], 0);
        nativeOptionValues[i] = env->GetStringUTFChars(jValueStrings[i], 0);
    }

    armnnDelegate::DelegateOptions delegateOptions(nativeOptionKeys,
                                                   nativeOptionValues,
                                                   numOptions,
                                                   nullptr);

    // Release jni memory. After the delegate options are created there is no need to hold on to it anymore.
    for (int i = 0; i < numOptions; i++)
    {
        env->ReleaseStringUTFChars(jKeyStrings[i], nativeOptionKeys[i]);
        env->ReleaseStringUTFChars(jValueStrings[i], nativeOptionValues[i]);
    }

    return reinterpret_cast<jlong>(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions));
}

/// Destroys a given Arm NN Delegate object
JNIEXPORT void
JNICALL Java_com_arm_armnn_delegate_ArmnnDelegate_deleteDelegate(JNIEnv* env, jclass clazz, jlong delegate)
{
    armnnDelegate::TfLiteArmnnDelegateDelete(reinterpret_cast<TfLiteDelegate*>(delegate));
}

/// Returns true if a Arm Mali GPU is detected.
/// Can be used to ensure that GpuAcc is supported on a device.
JNIEXPORT jboolean
JNICALL Java_com_arm_armnn_delegate_ArmnnUtils_IsGpuAccSupported(JNIEnv* env, jclass clazz)
{
#if defined(ARMCOMPUTECL_ENABLED)
    cl::Device device = cl::Device::getDefault();
    char device_name[32];
    cl_int err = clGetDeviceInfo(device.get(), CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
    if (err != CL_SUCCESS)
    {
        return false;
    }
    // search for "Mali" in the devices name
    if (strstr(device_name, "Mali"))
    {
        return true;
    }
#endif
    return false;
}

/// Returns true if the current device supports Neon instructions.
/// Can be used to ensure the CpuAcc backend is supported.
JNIEXPORT jboolean
JNICALL Java_com_arm_armnn_delegate_ArmnnUtils_IsNeonDetected(JNIEnv* env, jclass clazz)
{
    return armnn::NeonDetected();
}

}

