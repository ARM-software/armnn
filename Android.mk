#
# Copyright © 2017 ARM Ltd. All rights reserved.
# See LICENSE file in the project root for full license information.
#

LOCAL_PATH := $(call my-dir)

# Configure these paths if you move the source or Khronos headers
#
OPENCL_HEADER_PATH := $(LOCAL_PATH)/../clframework/include
NN_HEADER_PATH := $(LOCAL_PATH)/../../../../frameworks/ml/nn/runtime/include
ARMNN_HEADER_PATH := $(LOCAL_PATH)/include
ARMNN_SOURCE_HEADER_PATH := $(LOCAL_PATH)/src/armnn

include $(CLEAR_VARS)

LOCAL_C_INCLUDES :=  \
	$(OPENCL_HEADER_PATH) \
	$(NN_HEADER_PATH) \
	$(ARMNN_HEADER_PATH) \
	$(ARMNN_SOURCE_HEADER_PATH)

LOCAL_CFLAGS := \
	-std=c++14 \
	-fexceptions \
	-frtti \
	-isystem vendor/arm/android-nn-driver/boost_1_64_0 \
	-DARMCOMPUTECL_ENABLED \
	-DARMCOMPUTENEON_ENABLED

LOCAL_SRC_FILES :=  \
	src/armnn/test/UnitTests.cpp \
	src/armnn/test/EndToEndTest.cpp \
	src/armnn/test/UtilsTests.cpp \
	src/armnn/test/GraphTests.cpp \
	src/armnn/test/RuntimeTests.cpp \
	src/armnn/test/TensorTest.cpp \
	src/armnn/test/Network_test.cpp \
	src/armnn/backends/test/IsLayerSupportedTest.cpp \
	src/armnn/backends/test/Reference.cpp \
	src/armnn/backends/test/WorkloadDataValidation.cpp \
	src/armnn/backends/test/TensorCopyUtils.cpp \
	src/armnn/backends/test/LayerTests.cpp \
	src/armnn/backends/test/CreateWorkloadRef.cpp \
	src/armnn/backends/test/ArmComputeCl.cpp \
	src/armnn/backends/test/ArmComputeNeon.cpp \
	src/armnn/backends/test/CreateWorkloadCl.cpp \
	src/armnn/backends/test/CreateWorkloadNeon.cpp \
	src/armnn/backends/test/MemCopyTests.cpp

LOCAL_STATIC_LIBRARIES := \
	libneuralnetworks_common \
	libarmnn \
	libboost_log \
	libboost_system \
	libboost_unit_test_framework \
	libboost_thread \
	armnn-arm_compute

LOCAL_SHARED_LIBRARIES :=  \
	libbase \
	libhidlbase \
	libhidltransport \
	libhidlmemory \
	liblog \
	libutils \
	android.hardware.neuralnetworks@1.0 \
	android.hidl.allocator@1.0 \
	android.hidl.memory@1.0 \
	libOpenCL

LOCAL_MODULE := armnn-tests

LOCAL_MODULE_TAGS := eng optional

LOCAL_ARM_MODE := arm

# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_EXECUTABLE)



