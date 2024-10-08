#
# Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

set(armnnOpaqueDelegateObject_sources)
list(APPEND armnnOpaqueDelegateObject_sources
        include/armnn_delegate.hpp
        include/Version.hpp
        src/Activation.hpp
        src/ArgMinMax.hpp
        src/armnn_delegate.cpp
        src/armnn_external_delegate.cpp
        src/BatchMatMul.hpp
        src/BatchSpace.hpp
        src/BroadcastTo.hpp
        src/Comparison.hpp
        src/Control.hpp
        src/Convolution.hpp
        src/ElementwiseBinary.hpp
        src/ElementwiseUnary.hpp
        src/FullyConnected.hpp
        src/Gather.hpp
        src/GatherNd.hpp
        src/LogicalBinary.hpp
        src/Lstm.hpp
        src/Normalization.hpp
        src/Pad.hpp
        src/Pooling.hpp
        src/Pack.hpp
        src/Prelu.hpp
        src/Redefine.hpp
        src/Reduce.hpp
        src/Resize.hpp
        src/ReverseV2.hpp
        src/Round.hpp
        src/ScatterNd.hpp
        src/Shape.hpp
        src/SharedFunctions.cpp
        src/SharedFunctions.hpp
        src/Slice.hpp
        src/Softmax.hpp
        src/SpaceDepth.hpp
        src/Split.hpp
        src/StridedSlice.hpp
        src/Tile.hpp
        src/Transpose.hpp
        src/UnidirectionalSequenceLstm.hpp
        src/Unpack.hpp)

add_library(armnnOpaqueDelegateObject OBJECT ${armnnOpaqueDelegateObject_sources})

target_include_directories(armnnOpaqueDelegateObject
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
add_library(tflite_headers_opaque INTERFACE)
target_include_directories(tflite_headers_opaque
        INTERFACE
            $<BUILD_INTERFACE:${TfLite_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:include/tflite_headers_opaque>)

target_compile_options(tflite_headers_opaque 
        INTERFACE 
            -Wno-conversion
            -Wno-sign-conversion
            -Wno-unused-parameter
            -Wno-unused-function)

target_link_libraries(armnnOpaqueDelegateObject PUBLIC tflite_headers_opaque)

## Add Flatbuffers dependency
find_package(Flatbuffers REQUIRED MODULE)

target_link_libraries(armnnOpaqueDelegateObject PRIVATE ${Flatbuffers_LIB})

# include/flatbuffers/flatbuffers.h is not warning clean
# We can't change compilation flags on header files directly, so we need to add them to an interface library first
add_library(flatbuffer_headers_opaque INTERFACE)
target_include_directories(flatbuffer_headers_opaque 
        INTERFACE 
            $<BUILD_INTERFACE:${Flatbuffers_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:include/flatbuffer_headers_opaque>)
target_compile_options(flatbuffer_headers_opaque INTERFACE -Wno-sign-conversion -Wno-sign-compare)

target_link_libraries(armnnOpaqueDelegateObject PUBLIC flatbuffer_headers_opaque)

# Additional Absl Sync for Opaque Delegate
find_package(TfLiteAbsl REQUIRED MODULE)
target_include_directories(armnnOpaqueDelegateObject PUBLIC ${TfLite_ABSL_SYNC_HEADERS})
target_link_libraries(armnnOpaqueDelegateObject PUBLIC ${TfLite_Extra_Absl_LIB})

####################################################
## Export targets
install(TARGETS armnnOpaqueDelegateObject
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
