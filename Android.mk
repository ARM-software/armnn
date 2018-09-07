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
ARMNN_SOURCE_HEADER_PATH := $(LOCAL_PATH)/src/armnn
ARMNN_SOURCE_UTILS_HEADER_PATH := $(LOCAL_PATH)/src/armnnUtils

##############
# libarmnn.a #
##############
include $(CLEAR_VARS)

LOCAL_MODULE := libarmnn
LOCAL_MODULE_TAGS := eng optional
LOCAL_ARM_MODE := arm
LOCAL_PROPRIETARY_MODULE := true

# Mark source files as dependent on Android.mk
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk

LOCAL_EXPORT_C_INCLUDES := \
        $(ARMNN_SOURCE_HEADER_PATH) \
        $(ARMNN_SOURCE_UTILS_HEADER_PATH)

LOCAL_C_INCLUDES := \
	$(OPENCL_HEADER_PATH) \
	$(NN_HEADER_PATH) \
	$(ARMNN_HEADER_PATH) \
	$(ARMNN_SOURCE_HEADER_PATH) \
	$(ARMNN_SOURCE_UTILS_HEADER_PATH)

LOCAL_SRC_FILES := \
        src/armnnUtils/DotSerializer.cpp \
        src/armnnUtils/FloatingPointConverter.cpp \
        src/armnnUtils/Logging.cpp \
        src/armnnUtils/Permute.cpp \
        src/armnn/backends/ArmComputeTensorUtils.cpp \
        src/armnn/backends/ClWorkloads/ClActivationFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClActivationUint8Workload.cpp \
        src/armnn/backends/ClWorkloads/ClAdditionBaseWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClAdditionFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClAdditionUint8Workload.cpp \
        src/armnn/backends/ClWorkloads/ClSubtractionBaseWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClSubtractionFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClSubtractionUint8Workload.cpp \
        src/armnn/backends/ClWorkloads/ClBaseConstantWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClBatchNormalizationFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClConstantFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClConstantUint8Workload.cpp \
        src/armnn/backends/ClWorkloads/ClConvertFp16ToFp32Workload.cpp \
        src/armnn/backends/ClWorkloads/ClConvertFp32ToFp16Workload.cpp \
        src/armnn/backends/ClWorkloads/ClConvolution2dBaseWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClConvolution2dFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClConvolution2dUint8Workload.cpp \
        src/armnn/backends/ClWorkloads/ClDepthwiseConvolutionBaseWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClDepthwiseConvolutionFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClDepthwiseConvolutionUint8Workload.cpp \
        src/armnn/backends/ClWorkloads/ClDivisionFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClFloorFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClFullyConnectedFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClL2NormalizationFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClLstmFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClMergerFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClMergerUint8Workload.cpp \
        src/armnn/backends/ClWorkloads/ClMultiplicationFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClNormalizationFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClPermuteWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClPooling2dBaseWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClPooling2dFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClPooling2dUint8Workload.cpp \
        src/armnn/backends/ClWorkloads/ClReshapeFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClReshapeUint8Workload.cpp \
        src/armnn/backends/ClWorkloads/ClResizeBilinearFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClSoftmaxBaseWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClSoftmaxFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClSoftmaxUint8Workload.cpp \
        src/armnn/backends/ClWorkloads/ClSplitterFloatWorkload.cpp \
        src/armnn/backends/ClWorkloads/ClSplitterUint8Workload.cpp \
        src/armnn/backends/NeonWorkloads/NeonActivationFloatWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonActivationUint8Workload.cpp \
        src/armnn/backends/NeonWorkloads/NeonAdditionFloatWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonBatchNormalizationFloatWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonConstantFloatWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonConstantUint8Workload.cpp \
        src/armnn/backends/NeonWorkloads/NeonConvertFp16ToFp32Workload.cpp \
        src/armnn/backends/NeonWorkloads/NeonConvertFp32ToFp16Workload.cpp \
        src/armnn/backends/NeonWorkloads/NeonConvolution2dBaseWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonConvolution2dFloatWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonConvolution2dUint8Workload.cpp \
        src/armnn/backends/NeonWorkloads/NeonDepthwiseConvolutionBaseWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonDepthwiseConvolutionFloatWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonDepthwiseConvolutionUint8Workload.cpp \
        src/armnn/backends/NeonWorkloads/NeonFloorFloatWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonFullyConnectedFloatWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonL2NormalizationFloatWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonLstmFloatWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonMergerFloatWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonMergerUint8Workload.cpp \
        src/armnn/backends/NeonWorkloads/NeonMultiplicationFloatWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonNormalizationFloatWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonPermuteWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonPooling2dBaseWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonPooling2dFloatWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonPooling2dUint8Workload.cpp \
        src/armnn/backends/NeonWorkloads/NeonReshapeFloatWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonReshapeUint8Workload.cpp \
        src/armnn/backends/NeonWorkloads/NeonSoftmaxBaseWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonSoftmaxFloatWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonSoftmaxUint8Workload.cpp \
        src/armnn/backends/NeonWorkloads/NeonSplitterFloatWorkload.cpp \
        src/armnn/backends/NeonWorkloads/NeonSplitterUint8Workload.cpp \
        src/armnn/backends/ClWorkloadFactory.cpp \
        src/armnn/backends/ClContextControl.cpp \
        src/armnn/backends/CpuTensorHandle.cpp \
        src/armnn/backends/RefWorkloadFactory.cpp \
        src/armnn/backends/RefWorkloads/RefMergerUint8Workload.cpp \
        src/armnn/backends/RefWorkloads/RefResizeBilinearUint8Workload.cpp \
        src/armnn/backends/RefWorkloads/FullyConnected.cpp \
        src/armnn/backends/RefWorkloads/RefFullyConnectedFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/RefSoftmaxFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/RefActivationFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/RefBatchNormalizationUint8Workload.cpp \
        src/armnn/backends/RefWorkloads/Multiplication.cpp \
        src/armnn/backends/RefWorkloads/RefMultiplicationUint8Workload.cpp \
        src/armnn/backends/RefWorkloads/RefBaseConstantWorkload.cpp \
        src/armnn/backends/RefWorkloads/RefResizeBilinearFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/RefBatchNormalizationFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/Broadcast.cpp \
        src/armnn/backends/RefWorkloads/Addition.cpp \
        src/armnn/backends/RefWorkloads/RefAdditionUint8Workload.cpp \
        src/armnn/backends/RefWorkloads/RefAdditionFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/Subtraction.cpp \
        src/armnn/backends/RefWorkloads/RefSubtractionUint8Workload.cpp \
        src/armnn/backends/RefWorkloads/RefSubtractionFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/RefFakeQuantizationFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/ResizeBilinear.cpp \
        src/armnn/backends/RefWorkloads/RefSoftmaxUint8Workload.cpp \
        src/armnn/backends/RefWorkloads/RefDepthwiseConvolution2dFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/RefPooling2dUint8Workload.cpp \
        src/armnn/backends/RefWorkloads/RefFloorFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/ConvImpl.cpp \
        src/armnn/backends/RefWorkloads/Activation.cpp \
        src/armnn/backends/RefWorkloads/RefReshapeUint8Workload.cpp \
        src/armnn/backends/RefWorkloads/RefL2NormalizationFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/RefLstmFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/RefConvolution2dFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/RefConvolution2dUint8Workload.cpp \
        src/armnn/backends/RefWorkloads/RefSplitterFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/RefActivationUint8Workload.cpp \
        src/armnn/backends/RefWorkloads/RefSplitterUint8Workload.cpp \
        src/armnn/backends/RefWorkloads/RefPooling2dFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/RefMultiplicationFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/RefReshapeFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/RefNormalizationFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/Softmax.cpp \
        src/armnn/backends/RefWorkloads/RefDepthwiseConvolution2dUint8Workload.cpp \
        src/armnn/backends/RefWorkloads/RefConstantUint8Workload.cpp \
        src/armnn/backends/RefWorkloads/RefConstantFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/Pooling2d.cpp \
        src/armnn/backends/RefWorkloads/RefMergerFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/RefFullyConnectedUint8Workload.cpp \
        src/armnn/backends/RefWorkloads/RefPermuteWorkload.cpp \
        src/armnn/backends/RefWorkloads/RefConvertFp16ToFp32Workload.cpp \
        src/armnn/backends/RefWorkloads/RefConvertFp32ToFp16Workload.cpp \
        src/armnn/backends/RefWorkloads/Division.cpp \
        src/armnn/backends/RefWorkloads/RefDivisionFloat32Workload.cpp \
        src/armnn/backends/RefWorkloads/RefDivisionUint8Workload.cpp \
        src/armnn/backends/MemCopyWorkload.cpp \
        src/armnn/backends/WorkloadData.cpp \
        src/armnn/backends/WorkloadFactory.cpp \
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
        src/armnn/layers/MemCopyLayer.cpp \
        src/armnn/layers/MergerLayer.cpp \
        src/armnn/layers/MultiplicationLayer.cpp \
        src/armnn/layers/NormalizationLayer.cpp \
        src/armnn/layers/OutputLayer.cpp \
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
        src/armnn/backends/OutputHandler.cpp \
        src/armnn/OpenClTimer.cpp \
        src/armnn/WallClockTimer.cpp \
        src/armnn/ProfilingEvent.cpp \
        src/armnn/Profiling.cpp \
        src/armnn/JsonPrinter.cpp \
        src/armnn/Tensor.cpp \
        src/armnn/Utils.cpp \
        src/armnn/LayerSupport.cpp \
        src/armnn/Observable.cpp \
        src/armnn/backends/RefLayerSupport.cpp \
        src/armnn/backends/ClLayerSupport.cpp \
        src/armnn/backends/NeonLayerSupport.cpp \
        src/armnn/backends/NeonWorkloadUtils.cpp \
        src/armnn/backends/NeonWorkloadFactory.cpp \
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

