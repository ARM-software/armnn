#
# Copyright © 2017 ARM Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

LOCAL_PATH := $(call my-dir)

# Configure these paths if you move the source or Khronos headers
OPENCL_HEADER_PATH := $(LOCAL_PATH)/../clframework/include
NN_HEADER_PATH := $(LOCAL_PATH)/../../../../frameworks/ml/nn/runtime/include
ARMNN_HEADER_PATH := $(LOCAL_PATH)/include
ARMNN_MAIN_HEADER_PATH := $(LOCAL_PATH)/src
ARMNN_SOURCE_HEADER_PATH := $(LOCAL_PATH)/src/armnn
ARMNN_SOURCE_UTILS_HEADER_PATH := $(LOCAL_PATH)/src/armnnUtils
ARMNN_BACKENDS_HEADER_PATH := $(LOCAL_PATH)/src/backends
ARMNN_PROFILING_HEADER_PATH := $(LOCAL_PATH)/src/profiling

# find the common.mk and backend.mk files in the backend source folders
ARMNN_BACKEND_COMMON_MAKEFILE_LOCAL_PATHS := $(wildcard $(LOCAL_PATH)/src/backends/*/common.mk)
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
ifeq ($(Q_OR_LATER),1)
# "eng" is deprecated in Android Q
LOCAL_MODULE_TAGS := optional
else
LOCAL_MODULE_TAGS := eng optional
endif
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
        $(eval ARMNN_BACKEND_SOURCES := $(ARMNN_BACKEND_SOURCES)\
        $(patsubst %,$(mkPath)/%,$(COMMON_SOURCES))))

$(foreach mkPath,$(ARMNN_BACKEND_MAKEFILE_DIRS),\
        $(eval include $(LOCAL_PATH)/$(mkPath)/backend.mk)\
        $(eval ARMNN_BACKEND_SOURCES := $(ARMNN_BACKEND_SOURCES)\
        $(patsubst %,$(mkPath)/%,$(BACKEND_SOURCES))))

# Mark source files as dependent on Android.mk and backend makefiles
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk \
                                 $(ARMNN_BACKEND_MAKEFILE_LOCAL_PATHS)

LOCAL_EXPORT_C_INCLUDES := \
        $(ARMNN_MAIN_HEADER_PATH) \
        $(ARMNN_SOURCE_HEADER_PATH) \
        $(ARMNN_SOURCE_UTILS_HEADER_PATH) \
        $(ARMNN_PROFILING_HEADER_PATH) \
        $(ARMNN_BACKENDS_HEADER_PATH)

LOCAL_C_INCLUDES := \
        $(OPENCL_HEADER_PATH) \
        $(NN_HEADER_PATH) \
        $(ARMNN_HEADER_PATH) \
        $(ARMNN_MAIN_HEADER_PATH) \
        $(ARMNN_SOURCE_HEADER_PATH) \
        $(ARMNN_SOURCE_UTILS_HEADER_PATH) \
        $(ARMNN_PROFILING_HEADER_PATH) \
        $(ARMNN_BACKENDS_HEADER_PATH)

LOCAL_SRC_FILES := \
        $(ARMNN_BACKEND_SOURCES) \
        src/armnn/BackendHelper.cpp \
        src/armnn/BackendRegistry.cpp \
        src/armnn/Descriptors.cpp \
        src/armnn/Exceptions.cpp \
        src/armnn/Graph.cpp \
        src/armnn/InternalTypes.cpp \
        src/armnn/JsonPrinter.cpp \
        src/armnn/Layer.cpp \
        src/armnn/LayerSupport.cpp \
        src/armnn/LoadedNetwork.cpp \
        src/armnn/Logging.cpp \
        src/armnn/Network.cpp \
        src/armnn/NetworkUtils.cpp \
        src/armnn/Observable.cpp \
        src/armnn/Optimizer.cpp \
        src/armnn/optimizations/PermuteAndBatchToSpaceAsDepthToSpace.cpp \
        src/armnn/OutputHandler.cpp \
        src/armnn/ProfilingEvent.cpp \
        src/armnn/Profiling.cpp \
        src/armnn/Runtime.cpp \
        src/armnn/SerializeLayerParameters.cpp \
        src/armnn/SubgraphView.cpp \
        src/armnn/SubgraphViewSelector.cpp \
        src/armnn/Tensor.cpp \
        src/armnn/TypesUtils.cpp \
        src/armnn/Utils.cpp \
        src/armnn/WallClockTimer.cpp \
        src/armnnUtils/CsvReader.cpp \
        src/armnnUtils/DataLayoutIndexed.cpp \
        src/armnnUtils/DotSerializer.cpp \
        src/armnnUtils/FloatingPointConverter.cpp \
        src/armnnUtils/HeapProfiling.cpp \
        src/armnnUtils/LeakChecking.cpp \
        src/armnnUtils/ParserHelper.cpp \
        src/armnnUtils/Permute.cpp \
        src/armnnUtils/TensorUtils.cpp \
        src/armnnUtils/VerificationHelpers.cpp \
        src/armnnUtils/NetworkSockets.cpp \
        src/armnnUtils/Filesystem.cpp \
        src/armnnUtils/Processes.cpp \
        src/armnn/layers/ActivationLayer.cpp \
        src/armnn/layers/AdditionLayer.cpp \
        src/armnn/layers/ArgMinMaxLayer.cpp \
        src/armnn/layers/BatchNormalizationLayer.cpp \
        src/armnn/layers/BatchToSpaceNdLayer.cpp \
        src/armnn/layers/ComparisonLayer.cpp \
        src/armnn/layers/ConcatLayer.cpp \
        src/armnn/layers/ConstantLayer.cpp \
        src/armnn/layers/Convolution2dLayer.cpp \
        src/armnn/layers/ConvertFp16ToFp32Layer.cpp \
        src/armnn/layers/ConvertFp32ToFp16Layer.cpp \
        src/armnn/layers/DebugLayer.cpp \
        src/armnn/layers/DepthToSpaceLayer.cpp \
        src/armnn/layers/DepthwiseConvolution2dLayer.cpp \
        src/armnn/layers/DequantizeLayer.cpp \
        src/armnn/layers/DetectionPostProcessLayer.cpp \
        src/armnn/layers/DivisionLayer.cpp \
        src/armnn/layers/ElementwiseBaseLayer.cpp \
        src/armnn/layers/ElementwiseUnaryLayer.cpp \
        src/armnn/layers/FakeQuantizationLayer.cpp \
        src/armnn/layers/FloorLayer.cpp \
        src/armnn/layers/FullyConnectedLayer.cpp \
        src/armnn/layers/GatherLayer.cpp \
        src/armnn/layers/InputLayer.cpp \
        src/armnn/layers/InstanceNormalizationLayer.cpp \
        src/armnn/layers/L2NormalizationLayer.cpp \
        src/armnn/layers/LogSoftmaxLayer.cpp \
        src/armnn/layers/LstmLayer.cpp \
        src/armnn/layers/MaximumLayer.cpp \
        src/armnn/layers/MeanLayer.cpp \
        src/armnn/layers/MemCopyLayer.cpp \
        src/armnn/layers/MemImportLayer.cpp \
        src/armnn/layers/MergeLayer.cpp \
        src/armnn/layers/MinimumLayer.cpp \
        src/armnn/layers/MultiplicationLayer.cpp \
        src/armnn/layers/NormalizationLayer.cpp \
        src/armnn/layers/OutputLayer.cpp \
        src/armnn/layers/PadLayer.cpp \
        src/armnn/layers/PermuteLayer.cpp \
        src/armnn/layers/Pooling2dLayer.cpp \
        src/armnn/layers/PreCompiledLayer.cpp \
        src/armnn/layers/PreluLayer.cpp \
        src/armnn/layers/QuantizeLayer.cpp \
        src/armnn/layers/QuantizedLstmLayer.cpp \
        src/armnn/layers/ReshapeLayer.cpp \
        src/armnn/layers/ResizeLayer.cpp \
        src/armnn/layers/SliceLayer.cpp \
        src/armnn/layers/SoftmaxLayer.cpp \
        src/armnn/layers/SpaceToBatchNdLayer.cpp \
        src/armnn/layers/SpaceToDepthLayer.cpp \
        src/armnn/layers/SplitterLayer.cpp \
        src/armnn/layers/StackLayer.cpp \
        src/armnn/layers/StandInLayer.cpp \
        src/armnn/layers/StridedSliceLayer.cpp \
        src/armnn/layers/SubtractionLayer.cpp \
        src/armnn/layers/SwitchLayer.cpp \
        src/armnn/layers/TransposeConvolution2dLayer.cpp \
        src/profiling/BufferManager.cpp \
        src/profiling/CommandHandler.cpp \
        src/profiling/CommandHandlerFunctor.cpp \
        src/profiling/CommandHandlerKey.cpp \
        src/profiling/CommandHandlerRegistry.cpp \
        src/profiling/ConnectionAcknowledgedCommandHandler.cpp \
        src/profiling/CounterDirectory.cpp \
        src/profiling/CounterIdMap.cpp \
        src/profiling/DirectoryCaptureCommandHandler.cpp \
        src/profiling/FileOnlyProfilingConnection.cpp \
        src/profiling/Holder.cpp \
        src/profiling/LabelsAndEventClasses.cpp \
        src/profiling/PacketBuffer.cpp \
        src/profiling/PacketVersionResolver.cpp \
        src/profiling/PeriodicCounterCapture.cpp \
        src/profiling/PeriodicCounterSelectionCommandHandler.cpp \
        src/profiling/PerJobCounterSelectionCommandHandler.cpp \
        src/profiling/ProfilingConnectionDumpToFileDecorator.cpp \
        src/profiling/ProfilingConnectionFactory.cpp \
        src/profiling/ProfilingService.cpp \
        src/profiling/ProfilingStateMachine.cpp \
        src/profiling/ProfilingUtils.cpp \
        src/profiling/RegisterBackendCounters.cpp \
        src/profiling/RequestCounterDirectoryCommandHandler.cpp \
        src/profiling/SendCounterPacket.cpp \
        src/profiling/SendThread.cpp \
        src/profiling/SendTimelinePacket.cpp \
        src/profiling/SocketProfilingConnection.cpp \
        src/profiling/TimelinePacketWriterFactory.cpp \
        src/profiling/TimelineUtilityMethods.cpp \
        src/profiling/backends/BackendProfiling.cpp

LOCAL_STATIC_LIBRARIES := \
        arm_compute_library \
        libboost_log \
        libboost_system \
        libboost_thread \
        libboost_filesystem

LOCAL_SHARED_LIBRARIES := \
        liblog

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions \
        -Wno-unused-parameter \
        -frtti

# The variable to enable/disable the CL backend (ARMNN_COMPUTE_CL_ENABLED) is declared in android-nn-driver/Android.mk
ifeq ($(ARMNN_COMPUTE_CL_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMCOMPUTECL_ENABLED
endif # ARMNN_COMPUTE_CL_ENABLED == 1
# The variable to enable/disable the NEON backend (ARMNN_COMPUTE_NEON_ENABLED) is declared in android-nn-driver/Android.mk
ifeq ($(ARMNN_COMPUTE_NEON_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMCOMPUTENEON_ENABLED
endif # ARMNN_COMPUTE_NEON_ENABLED == 1
# The variable to enable/disable the REFERENCE backend (ARMNN_REF_ENABLED) is declared in android-nn-driver/Android.mk
ifeq ($(ARMNN_REF_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMNNREF_ENABLED
endif # ARMNN_REF_ENABLED == 1

ifeq ($(Q_OR_LATER),1)
LOCAL_CFLAGS += \
        -DARMNN_MIXED_PRECISION_FP16_POOLING \
        -DBOOST_NO_AUTO_PTR
endif # PLATFORM_VERSION == Q or later

include $(BUILD_STATIC_LIBRARY)

###############
# armnn-tests #
###############
include $(CLEAR_VARS)

LOCAL_MODULE := armnn-tests
ifeq ($(Q_OR_LATER),1)
# "eng" is deprecated in Android Q
LOCAL_MODULE_TAGS := optional
else
LOCAL_MODULE_TAGS := eng optional
endif
LOCAL_ARM_MODE := arm
LOCAL_PROPRIETARY_MODULE := true

# placeholder to hold all backend unit test source files
ARMNN_BACKEND_TEST_SOURCES :=

#
# iterate through the backend common and specific include paths, include them into the current
# makefile and append the sources held by the COMMON_TEST_SOURCES and BACKEND_TEST_SOURCES
# (included from the given makefile) to the ARMNN_BACKEND_TEST_SOURCES list
#
$(foreach mkPath,$(ARMNN_BACKEND_COMMON_MAKEFILE_DIRS), \
        $(eval include $(LOCAL_PATH)/$(mkPath)/common.mk) \
        $(eval ARMNN_BACKEND_TEST_SOURCES := $(ARMNN_BACKEND_TEST_SOURCES) \
        $(patsubst %,$(mkPath)/%,$(COMMON_TEST_SOURCES))))

$(foreach mkPath,$(ARMNN_BACKEND_MAKEFILE_DIRS), \
        $(eval include $(LOCAL_PATH)/$(mkPath)/backend.mk) \
        $(eval ARMNN_BACKEND_TEST_SOURCES := $(ARMNN_BACKEND_TEST_SOURCES) \
        $(patsubst %,$(mkPath)/%,$(BACKEND_TEST_SOURCES))))

# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_C_INCLUDES := \
        $(OPENCL_HEADER_PATH) \
        $(NN_HEADER_PATH) \
        $(ARMNN_HEADER_PATH) \
        $(ARMNN_MAIN_HEADER_PATH) \
        $(ARMNN_SOURCE_HEADER_PATH) \
        $(ARMNN_SOURCE_UTILS_HEADER_PATH) \
        $(ARMNN_PROFILING_HEADER_PATH) \
        $(ARMNN_BACKENDS_HEADER_PATH)

LOCAL_CFLAGS := \
        -std=$(CPP_VERSION) \
        -fexceptions \
        -frtti \
        -isystem vendor/arm/android-nn-driver/boost_1_64_0

# The variable to enable/disable the CL backend (ARMNN_COMPUTE_CL_ENABLED) is declared in android-nn-driver/Android.mk
ifeq ($(ARMNN_COMPUTE_CL_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMCOMPUTECL_ENABLED
endif # ARMNN_COMPUTE_CL_ENABLED == 1
# The variable to enable/disable the NEON backend (ARMNN_COMPUTE_NEON_ENABLED) is declared in android-nn-driver/Android.mk
ifeq ($(ARMNN_COMPUTE_NEON_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMCOMPUTENEON_ENABLED
endif # ARMNN_COMPUTE_NEON_ENABLED == 1
# The variable to enable/disable the REFERENCE backend (ARMNN_REF_ENABLED) is declared in android-nn-driver/Android.mk
ifeq ($(ARMNN_REF_ENABLED),1)
LOCAL_CFLAGS += \
        -DARMNNREF_ENABLED
endif # ARMNN_REF_ENABLED == 1

ifeq ($(Q_OR_LATER),1)
LOCAL_CFLAGS += \
        -DBOOST_NO_AUTO_PTR
endif # PLATFORM_VERSION == Q or later

LOCAL_SRC_FILES := \
        $(ARMNN_BACKEND_TEST_SOURCES) \
        src/armnn/test/EndToEndTest.cpp \
        src/armnn/test/GraphTests.cpp \
        src/armnn/test/GraphUtils.cpp \
        src/armnn/test/InferOutputTests.cpp \
        src/armnn/test/InstrumentTests.cpp \
        src/armnn/test/NetworkTests.cpp \
        src/armnn/test/ObservableTest.cpp \
        src/armnn/test/OptionalTest.cpp \
        src/armnn/test/ProfilingEventTest.cpp \
        src/armnn/test/SubgraphViewTests.cpp \
        src/armnn/test/TensorHandleStrategyTest.cpp \
        src/armnn/test/TensorTest.cpp \
        src/armnn/test/TestUtils.cpp \
        src/armnn/test/UnitTests.cpp \
        src/armnn/test/UtilsTests.cpp \
        src/profiling/test/ProfilingTestUtils.cpp

ifeq ($(ARMNN_REF_ENABLED),1)
LOCAL_SRC_FILES += \
        src/armnn/test/DebugCallbackTest.cpp \
        src/armnn/test/RuntimeTests.cpp
endif

LOCAL_STATIC_LIBRARIES := \
        libneuralnetworks_common \
        libboost_log \
        libboost_system \
        libboost_unit_test_framework \
        libboost_thread \
        libboost_filesystem \
        arm_compute_library

LOCAL_WHOLE_STATIC_LIBRARIES := libarmnn

LOCAL_SHARED_LIBRARIES := \
        libbase \
        libhidlbase \
        libhidltransport \
        libhidlmemory \
        liblog \
        libutils \
        android.hardware.neuralnetworks@1.0 \
        android.hidl.allocator@1.0 \
        android.hidl.memory@1.0

ifeq ($(ARMNN_COMPUTE_CL_ENABLED),1)
LOCAL_SHARED_LIBRARIES += \
        libOpenCL
endif

include $(BUILD_EXECUTABLE)
