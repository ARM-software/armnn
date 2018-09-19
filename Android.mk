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
        src/armnnUtils/DotSerializer.cpp \
        src/armnnUtils/FloatingPointConverter.cpp \
        src/armnnUtils/Logging.cpp \
        src/armnnUtils/Permute.cpp \
        src/backends/ArmComputeTensorUtils.cpp \
        src/backends/ClWorkloads/ClActivationFloatWorkload.cpp \
        src/backends/ClWorkloads/ClActivationUint8Workload.cpp \
        src/backends/ClWorkloads/ClAdditionWorkload.cpp \
        src/backends/ClWorkloads/ClSubtractionWorkload.cpp \
        src/backends/ClWorkloads/ClBaseConstantWorkload.cpp \
        src/backends/ClWorkloads/ClBatchNormalizationFloatWorkload.cpp \
        src/backends/ClWorkloads/ClConstantFloatWorkload.cpp \
        src/backends/ClWorkloads/ClConstantUint8Workload.cpp \
        src/backends/ClWorkloads/ClConvertFp16ToFp32Workload.cpp \
        src/backends/ClWorkloads/ClConvertFp32ToFp16Workload.cpp \
        src/backends/ClWorkloads/ClConvolution2dBaseWorkload.cpp \
        src/backends/ClWorkloads/ClConvolution2dFloatWorkload.cpp \
        src/backends/ClWorkloads/ClConvolution2dUint8Workload.cpp \
        src/backends/ClWorkloads/ClDepthwiseConvolutionBaseWorkload.cpp \
        src/backends/ClWorkloads/ClDepthwiseConvolutionFloatWorkload.cpp \
        src/backends/ClWorkloads/ClDepthwiseConvolutionUint8Workload.cpp \
        src/backends/ClWorkloads/ClDivisionFloatWorkload.cpp \
        src/backends/ClWorkloads/ClFloorFloatWorkload.cpp \
        src/backends/ClWorkloads/ClFullyConnectedWorkload.cpp \
        src/backends/ClWorkloads/ClL2NormalizationFloatWorkload.cpp \
        src/backends/ClWorkloads/ClLstmFloatWorkload.cpp \
        src/backends/ClWorkloads/ClMergerFloatWorkload.cpp \
        src/backends/ClWorkloads/ClMergerUint8Workload.cpp \
        src/backends/ClWorkloads/ClMultiplicationFloatWorkload.cpp \
        src/backends/ClWorkloads/ClNormalizationFloatWorkload.cpp \
        src/backends/ClWorkloads/ClPermuteWorkload.cpp \
        src/backends/ClWorkloads/ClPooling2dBaseWorkload.cpp \
        src/backends/ClWorkloads/ClPooling2dFloatWorkload.cpp \
        src/backends/ClWorkloads/ClPooling2dUint8Workload.cpp \
        src/backends/ClWorkloads/ClReshapeFloatWorkload.cpp \
        src/backends/ClWorkloads/ClReshapeUint8Workload.cpp \
        src/backends/ClWorkloads/ClResizeBilinearFloatWorkload.cpp \
        src/backends/ClWorkloads/ClSoftmaxBaseWorkload.cpp \
        src/backends/ClWorkloads/ClSoftmaxFloatWorkload.cpp \
        src/backends/ClWorkloads/ClSoftmaxUint8Workload.cpp \
        src/backends/ClWorkloads/ClSplitterFloatWorkload.cpp \
        src/backends/ClWorkloads/ClSplitterUint8Workload.cpp \
        src/backends/NeonWorkloads/NeonActivationFloatWorkload.cpp \
        src/backends/NeonWorkloads/NeonActivationUint8Workload.cpp \
        src/backends/NeonWorkloads/NeonAdditionFloatWorkload.cpp \
        src/backends/NeonWorkloads/NeonBatchNormalizationFloatWorkload.cpp \
        src/backends/NeonWorkloads/NeonConstantFloatWorkload.cpp \
        src/backends/NeonWorkloads/NeonConstantUint8Workload.cpp \
        src/backends/NeonWorkloads/NeonConvertFp16ToFp32Workload.cpp \
        src/backends/NeonWorkloads/NeonConvertFp32ToFp16Workload.cpp \
        src/backends/NeonWorkloads/NeonConvolution2dBaseWorkload.cpp \
        src/backends/NeonWorkloads/NeonConvolution2dFloatWorkload.cpp \
        src/backends/NeonWorkloads/NeonConvolution2dUint8Workload.cpp \
        src/backends/NeonWorkloads/NeonDepthwiseConvolutionBaseWorkload.cpp \
        src/backends/NeonWorkloads/NeonDepthwiseConvolutionFloatWorkload.cpp \
        src/backends/NeonWorkloads/NeonDepthwiseConvolutionUint8Workload.cpp \
        src/backends/NeonWorkloads/NeonFloorFloatWorkload.cpp \
        src/backends/NeonWorkloads/NeonFullyConnectedFloatWorkload.cpp \
        src/backends/NeonWorkloads/NeonL2NormalizationFloatWorkload.cpp \
        src/backends/NeonWorkloads/NeonLstmFloatWorkload.cpp \
        src/backends/NeonWorkloads/NeonMergerFloatWorkload.cpp \
        src/backends/NeonWorkloads/NeonMergerUint8Workload.cpp \
        src/backends/NeonWorkloads/NeonMultiplicationFloatWorkload.cpp \
        src/backends/NeonWorkloads/NeonNormalizationFloatWorkload.cpp \
        src/backends/NeonWorkloads/NeonPermuteWorkload.cpp \
        src/backends/NeonWorkloads/NeonPooling2dBaseWorkload.cpp \
        src/backends/NeonWorkloads/NeonPooling2dFloatWorkload.cpp \
        src/backends/NeonWorkloads/NeonPooling2dUint8Workload.cpp \
        src/backends/NeonWorkloads/NeonReshapeFloatWorkload.cpp \
        src/backends/NeonWorkloads/NeonReshapeUint8Workload.cpp \
        src/backends/NeonWorkloads/NeonSoftmaxBaseWorkload.cpp \
        src/backends/NeonWorkloads/NeonSoftmaxFloatWorkload.cpp \
        src/backends/NeonWorkloads/NeonSoftmaxUint8Workload.cpp \
        src/backends/NeonWorkloads/NeonSplitterFloatWorkload.cpp \
        src/backends/NeonWorkloads/NeonSplitterUint8Workload.cpp \
        src/backends/NeonWorkloads/NeonSubtractionFloatWorkload.cpp \
        src/backends/ClWorkloadFactory.cpp \
        src/backends/ClContextControl.cpp \
        src/backends/CpuTensorHandle.cpp \
        src/backends/RefWorkloadFactory.cpp \
        src/backends/RefWorkloads/RefMergerUint8Workload.cpp \
        src/backends/RefWorkloads/RefResizeBilinearUint8Workload.cpp \
        src/backends/RefWorkloads/FullyConnected.cpp \
        src/backends/RefWorkloads/RefFullyConnectedFloat32Workload.cpp \
        src/backends/RefWorkloads/RefSoftmaxFloat32Workload.cpp \
        src/backends/RefWorkloads/RefActivationFloat32Workload.cpp \
        src/backends/RefWorkloads/RefBatchNormalizationUint8Workload.cpp \
        src/backends/RefWorkloads/RefBaseConstantWorkload.cpp \
        src/backends/RefWorkloads/RefResizeBilinearFloat32Workload.cpp \
        src/backends/RefWorkloads/RefBatchNormalizationFloat32Workload.cpp \
        src/backends/RefWorkloads/Broadcast.cpp \
        src/backends/RefWorkloads/ArithmeticFunction.cpp \
        src/backends/RefWorkloads/RefArithmeticWorkload.cpp \
        src/backends/RefWorkloads/RefFakeQuantizationFloat32Workload.cpp \
        src/backends/RefWorkloads/ResizeBilinear.cpp \
        src/backends/RefWorkloads/RefSoftmaxUint8Workload.cpp \
        src/backends/RefWorkloads/RefDepthwiseConvolution2dFloat32Workload.cpp \
        src/backends/RefWorkloads/RefPooling2dUint8Workload.cpp \
        src/backends/RefWorkloads/RefFloorFloat32Workload.cpp \
        src/backends/RefWorkloads/ConvImpl.cpp \
        src/backends/RefWorkloads/Activation.cpp \
        src/backends/RefWorkloads/RefReshapeUint8Workload.cpp \
        src/backends/RefWorkloads/RefL2NormalizationFloat32Workload.cpp \
        src/backends/RefWorkloads/RefLstmFloat32Workload.cpp \
        src/backends/RefWorkloads/RefConvolution2dFloat32Workload.cpp \
        src/backends/RefWorkloads/RefConvolution2dUint8Workload.cpp \
        src/backends/RefWorkloads/RefSplitterFloat32Workload.cpp \
        src/backends/RefWorkloads/RefActivationUint8Workload.cpp \
        src/backends/RefWorkloads/RefSplitterUint8Workload.cpp \
        src/backends/RefWorkloads/RefPooling2dFloat32Workload.cpp \
        src/backends/RefWorkloads/RefReshapeFloat32Workload.cpp \
        src/backends/RefWorkloads/RefNormalizationFloat32Workload.cpp \
        src/backends/RefWorkloads/Softmax.cpp \
        src/backends/RefWorkloads/RefDepthwiseConvolution2dUint8Workload.cpp \
        src/backends/RefWorkloads/RefConstantUint8Workload.cpp \
        src/backends/RefWorkloads/RefConstantFloat32Workload.cpp \
        src/backends/RefWorkloads/Pooling2d.cpp \
        src/backends/RefWorkloads/RefMergerFloat32Workload.cpp \
        src/backends/RefWorkloads/RefFullyConnectedUint8Workload.cpp \
        src/backends/RefWorkloads/RefPermuteWorkload.cpp \
        src/backends/RefWorkloads/RefConvertFp16ToFp32Workload.cpp \
        src/backends/RefWorkloads/RefConvertFp32ToFp16Workload.cpp \
        src/backends/MemCopyWorkload.cpp \
        src/backends/WorkloadData.cpp \
        src/backends/WorkloadFactory.cpp \
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
        src/backends/OutputHandler.cpp \
        src/armnn/OpenClTimer.cpp \
        src/armnn/WallClockTimer.cpp \
        src/armnn/ProfilingEvent.cpp \
        src/armnn/Profiling.cpp \
        src/armnn/JsonPrinter.cpp \
        src/armnn/Tensor.cpp \
        src/armnn/Utils.cpp \
        src/armnn/LayerSupport.cpp \
        src/armnn/Observable.cpp \
        src/backends/StringMapping.cpp \
        src/backends/RefLayerSupport.cpp \
        src/backends/ClLayerSupport.cpp \
        src/backends/NeonLayerSupport.cpp \
        src/backends/NeonWorkloadUtils.cpp \
        src/backends/NeonWorkloadFactory.cpp \
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

