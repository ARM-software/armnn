#
# Copyright © 2017 ARM Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

LOCAL_PATH := $(call my-dir)

# Configure these paths if you move the source or Khronos headers
#
OPENCL_HEADER_PATH := $(LOCAL_PATH)/../clframework/include
NN_HEADER_PATH := $(LOCAL_PATH)/../../../../frameworks/ml/nn/runtime/include
ARMNN_HEADER_PATH := $(LOCAL_PATH)/include
ARMNN_MAIN_HEADER_PATH := $(LOCAL_PATH)/src
ARMNN_SOURCE_HEADER_PATH := $(LOCAL_PATH)/src/armnn
ARMNN_SOURCE_UTILS_HEADER_PATH := $(LOCAL_PATH)/src/armnnUtils

# find the common.mk and backend.mk files in the backend source folders
ARMNN_BACKEND_COMMON_MAKEFILE_LOCAL_PATHS := $(wildcard $(LOCAL_PATH)/src/backends/*/common.mk) \
                                             $(LOCAL_PATH)/src/backends/common.mk
ARMNN_BACKEND_COMMON_MAKEFILE_PATHS := $(subst $(LOCAL_PATH),,$(ARMNN_BACKEND_COMMON_MAKEFILE_LOCAL_PATHS))
ARMNN_BACKEND_COMMON_MAKEFILE_DIRS := $(subst /common.mk,,$(ARMNN_BACKEND_COMMON_MAKEFILE_PATHS))

ARMNN_BACKEND_MAKEFILE_LOCAL_PATHS := $(wildcard $(LOCAL_PATH)/src/backends/*/backend.mk)
ARMNN_BACKEND_MAKEFILE_PATHS := $(subst $(LOCAL_PATH),,$(ARMNN_BACKEND_MAKEFILE_LOCAL_PATHS))
ARMNN_BACKEND_MAKEFILE_DIRS := $(subst /backend.mk,,$(ARMNN_BACKEND_MAKEFILE_PATHS))

##############
# libarmnn.a #
##############
include $(CLEAR_VARS)

LOCAL_MODULE := libarmnn
LOCAL_MODULE_TAGS := eng optional
LOCAL_ARM_MODE := arm
LOCAL_PROPRIETARY_MODULE := true

# placeholder to hold all backend source files, common and specific to the backends
ARMNN_BACKEND_SOURCES :=

#
# iterate through the backend common and specific include paths, include them into the
# current makefile and append the sources held by the COMMON_SOURCES and BACKEND_SOURCES variable
# (included from the given makefile) to the ARMNN_BACKEND_SOURCES list
#
$(foreach mkPath,$(ARMNN_BACKEND_COMMON_MAKEFILE_DIRS),\
   $(eval include $(LOCAL_PATH)/$(mkPath)/common.mk)\
   $(eval ARMNN_BACKEND_SOURCES := $(ARMNN_BACKEND_SOURCES) $(patsubst %,$(mkPath)/%,$(COMMON_SOURCES))))

$(foreach mkPath,$(ARMNN_BACKEND_MAKEFILE_DIRS),\
   $(eval include $(LOCAL_PATH)/$(mkPath)/backend.mk)\
   $(eval ARMNN_BACKEND_SOURCES := $(ARMNN_BACKEND_SOURCES) $(patsubst %,$(mkPath)/%,$(BACKEND_SOURCES))))

# Mark source files as dependent on Android.mk and backend makefiles
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk \
                                 $(ARMNN_BACKEND_MAKEFILE_LOCAL_PATHS)

LOCAL_EXPORT_C_INCLUDES := \
        $(ARMNN_MAIN_HEADER_PATH) \
        $(ARMNN_SOURCE_HEADER_PATH) \
        $(ARMNN_SOURCE_UTILS_HEADER_PATH)

LOCAL_C_INCLUDES := \
	$(OPENCL_HEADER_PATH) \
	$(NN_HEADER_PATH) \
	$(ARMNN_HEADER_PATH) \
	$(ARMNN_MAIN_HEADER_PATH) \
	$(ARMNN_SOURCE_HEADER_PATH) \
	$(ARMNN_SOURCE_UTILS_HEADER_PATH)

LOCAL_SRC_FILES := \
        $(ARMNN_BACKEND_SOURCES) \
        src/armnnUtils/DotSerializer.cpp \
        src/armnnUtils/FloatingPointConverter.cpp \
        src/armnnUtils/Logging.cpp \
        src/armnnUtils/Permute.cpp \
        src/armnnUtils/ParserHelper.cpp \
        src/armnn/layers/ActivationLayer.cpp \
        src/armnn/layers/AdditionLayer.cpp \
        src/armnn/layers/ArithmeticBaseLayer.cpp \
        src/armnn/layers/BatchNormalizationLayer.cpp \
        src/armnn/layers/ConstantLayer.cpp \
        src/armnn/layers/Convolution2dLayer.cpp \
        src/armnn/layers/ConvertFp16ToFp32Layer.cpp \
        src/armnn/layers/ConvertFp32ToFp16Layer.cpp \
        src/armnn/layers/DepthwiseConvolution2dLayer.cpp \
        src/armnn/layers/FakeQuantizationLayer.cpp \
        src/armnn/layers/FloorLayer.cpp \
        src/armnn/layers/FullyConnectedLayer.cpp \
        src/armnn/layers/InputLayer.cpp \
        src/armnn/layers/L2NormalizationLayer.cpp \
        src/armnn/layers/LstmLayer.cpp \
        src/armnn/layers/MeanLayer.cpp \
        src/armnn/layers/MemCopyLayer.cpp \
        src/armnn/layers/MergerLayer.cpp \
        src/armnn/layers/MultiplicationLayer.cpp \
        src/armnn/layers/NormalizationLayer.cpp \
        src/armnn/layers/OutputLayer.cpp \
        src/armnn/layers/PadLayer.cpp \
        src/armnn/layers/PermuteLayer.cpp \
        src/armnn/layers/Pooling2dLayer.cpp \
        src/armnn/layers/DivisionLayer.cpp \
        src/armnn/layers/SubtractionLayer.cpp \
        src/armnn/layers/ReshapeLayer.cpp \
        src/armnn/layers/ResizeBilinearLayer.cpp \
        src/armnn/layers/SoftmaxLayer.cpp \
        src/armnn/layers/SplitterLayer.cpp \
        src/armnn/Descriptors.cpp \
        src/armnn/Exceptions.cpp \
        src/armnn/Graph.cpp \
        src/armnn/Optimizer.cpp \
        src/armnn/Runtime.cpp \
        src/armnn/SerializeLayerParameters.cpp \
        src/armnn/InternalTypes.cpp \
        src/armnn/Layer.cpp \
        src/armnn/LoadedNetwork.cpp \
        src/armnn/NeonInterceptorScheduler.cpp \
        src/armnn/NeonTimer.cpp \
        src/armnn/Network.cpp \
        src/armnn/OpenClTimer.cpp \
        src/armnn/WallClockTimer.cpp \
        src/armnn/ProfilingEvent.cpp \
        src/armnn/Profiling.cpp \
        src/armnn/JsonPrinter.cpp \
        src/armnn/Tensor.cpp \
        src/armnn/Utils.cpp \
        src/armnn/LayerSupport.cpp \
        src/armnn/Observable.cpp \
        src/armnn/memory/BaseMemoryManager.cpp \
        src/armnn/memory/BlobLifetimeManager.cpp \
        src/armnn/memory/BlobMemoryPool.cpp \
        src/armnn/memory/OffsetLifetimeManager.cpp \
        src/armnn/memory/OffsetMemoryPool.cpp \
        src/armnn/memory/PoolManager.cpp

LOCAL_STATIC_LIBRARIES := \
	armnn-arm_compute \
        libboost_log \
        libboost_system \
        libboost_thread

LOCAL_SHARED_LIBRARIES := \
        liblog

LOCAL_CFLAGS := \
        -std=c++14 \
        -fexceptions \
        -DARMCOMPUTECL_ENABLED \
        -DARMCOMPUTENEON_ENABLED \
        -Wno-unused-parameter \
        -frtti

include $(BUILD_STATIC_LIBRARY)

###############
# armnn-tests #
###############
include $(CLEAR_VARS)

LOCAL_MODULE := armnn-tests
LOCAL_MODULE_TAGS := eng optional
LOCAL_ARM_MODE := arm
LOCAL_PROPRIETARY_MODULE := true

# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES := \
	$(OPENCL_HEADER_PATH) \
	$(NN_HEADER_PATH) \
	$(ARMNN_HEADER_PATH) \
	$(ARMNN_MAIN_HEADER_PATH) \
	$(ARMNN_SOURCE_HEADER_PATH) \
	$(ARMNN_SOURCE_UTILS_HEADER_PATH)

LOCAL_CFLAGS := \
	-std=c++14 \
	-fexceptions \
	-frtti \
	-isystem vendor/arm/android-nn-driver/boost_1_64_0 \
	-DARMCOMPUTECL_ENABLED \
	-DARMCOMPUTENEON_ENABLED

LOCAL_SRC_FILES := \
	src/armnn/test/UnitTests.cpp \
	src/armnn/test/EndToEndTest.cpp \
	src/armnn/test/UtilsTests.cpp \
	src/armnn/test/GraphTests.cpp \
	src/armnn/test/RuntimeTests.cpp \
	src/armnn/test/TensorTest.cpp \
	src/armnn/test/NeonTimerTest.cpp \
	src/armnn/test/NetworkTests.cpp \
	src/armnn/test/InstrumentTests.cpp \
	src/armnn/test/OpenClTimerTest.cpp \
	src/armnn/test/ProfilingEventTest.cpp \
	src/armnn/test/ObservableTest.cpp \
	src/armnn/test/OptionalTest.cpp \
	src/backends/test/IsLayerSupportedTest.cpp \
	src/backends/test/Reference.cpp \
	src/backends/test/WorkloadDataValidation.cpp \
	src/backends/test/TensorCopyUtils.cpp \
	src/backends/test/LayerTests.cpp \
	src/backends/test/CreateWorkloadRef.cpp \
	src/backends/test/ArmComputeCl.cpp \
	src/backends/test/ArmComputeNeon.cpp \
	src/backends/test/CreateWorkloadCl.cpp \
	src/backends/test/CreateWorkloadNeon.cpp \
	src/backends/test/MemCopyTests.cpp

LOCAL_STATIC_LIBRARIES := \
	libneuralnetworks_common \
	libarmnn \
	libboost_log \
	libboost_system \
	libboost_unit_test_framework \
	libboost_thread \
	armnn-arm_compute

LOCAL_SHARED_LIBRARIES := \
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

include $(BUILD_EXECUTABLE)

