#
# Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

set(armnnClassicDelegateObject_sources)
list(APPEND armnnClassicDelegateObject_sources
        include/armnn_delegate.hpp
        include/Version.hpp
        src/armnn_delegate.cpp
        src/armnn_external_delegate.cpp
        src/Activation.hpp
        src/ArgMinMax.hpp
        src/BatchMatMul.hpp
        src/BatchSpace.hpp
        src/ClassicDelegateUtils.hpp
        src/Comparison.hpp
        src/Convolution.hpp
        src/Control.hpp
        src/ElementwiseBinary.hpp
        src/ElementwiseUnary.hpp
        src/Fill.hpp
        src/FullyConnected.hpp
        src/Gather.hpp
        src/GatherNd.hpp
        src/LogicalBinary.hpp
        src/Lstm.hpp
        src/MultiLayerFacade.hpp
        src/Normalization.hpp
        src/Pack.hpp
        src/Pad.hpp
        src/Pooling.hpp
        src/Prelu.hpp
        src/Quantization.hpp
        src/Redefine.hpp
        src/Reduce.hpp
        src/Resize.hpp
        src/Round.hpp
        src/Shape.hpp
        src/SharedFunctions.hpp
        src/SharedFunctions.cpp
        src/Slice.hpp
        src/Softmax.hpp
        src/SpaceDepth.hpp
        src/Split.hpp
        src/Unpack.hpp
        src/Transpose.hpp)

add_library(armnnClassicDelegateObject OBJECT ${armnnClassicDelegateObject_sources})

target_include_directories(armnnClassicDelegateObject
        PUBLIC
            $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/common/include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/common/src)

## Add Tflite dependency
if(NOT TfLite_INCLUDE_DIR OR NOT TfLite_Schema_INCLUDE_PATH)
    find_package(TfLiteSrc REQUIRED MODULE)
endif()

# Various tflite header files are not warning clean
# We can't change compilation flags on header files directly, so we need to add them to an interface library first
add_library(tflite_headers INTERFACE)
target_include_directories(tflite_headers
        INTERFACE
            $<BUILD_INTERFACE:${TfLite_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:include/tflite_headers>)

target_compile_options(tflite_headers
        INTERFACE
            -Wno-conversion
            -Wno-sign-conversion
            -Wno-unused-parameter
            -Wno-unused-function)

target_link_libraries(armnnClassicDelegateObject PUBLIC tflite_headers)

## Add Flatbuffers dependency
find_package(Flatbuffers REQUIRED MODULE)

target_link_libraries(armnnClassicDelegateObject PRIVATE ${Flatbuffers_LIB})

# include/flatbuffers/flatbuffers.h is not warning clean
# We can't change compilation flags on header files directly, so we need to add them to an interface library first
add_library(flatbuffer_headers INTERFACE)
target_include_directories(flatbuffer_headers
        INTERFACE
            $<BUILD_INTERFACE:${Flatbuffers_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:include/flatbuffer_headers>)
target_compile_options(flatbuffer_headers INTERFACE -Wno-sign-conversion)

target_link_libraries(armnnClassicDelegateObject PUBLIC flatbuffer_headers)

####################################################
## Export targets
install(TARGETS armnnClassicDelegateObject
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})