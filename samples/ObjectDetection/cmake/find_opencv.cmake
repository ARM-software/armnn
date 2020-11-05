# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

set(OPENCV_VERSION 4.0.0)
set(FFMPEG_VERSION 4.2.1)
set(LIBX264_VERSION stable)

set(OPENCV_LIB OpenCV${OPENCV_VERSION})
set(FFMPEG_LIB ffmpeg${FFMPEG_VERSION})
set(X264_LIB   x264${LIBX264_VERSION})

set(OPENCV_NAMES
    libopencv_core.so.${OPENCV_VERSION}
    libopencv_imgproc.so.${OPENCV_VERSION}
    libopencv_imgcodecs.so.${OPENCV_VERSION}
    libopencv_videoio.so.${OPENCV_VERSION}
    libopencv_video.so.${OPENCV_VERSION}
    libopencv_highgui.so.${OPENCV_VERSION})

set(OPENCV_LIBS)
set(FFMPEG_LIBS)

foreach(opencv_lib ${OPENCV_NAMES})
    find_library(OPENCV_${opencv_lib}
        NAMES
            ${opencv_lib}
        HINTS
            ${OPENCV_LIB_DIR}
        PATHS
            ${OPENCV_LIB_DIR}
        PATH_SUFFIXES
            "lib"
            "lib64")
    if(OPENCV_${opencv_lib})
        message("Found library ${OPENCV_${opencv_lib}}")
        list(APPEND OPENCV_LIBS ${OPENCV_${opencv_lib}})
        get_filename_component(OPENCV_LIB_DIR ${OPENCV_${opencv_lib}} DIRECTORY)
        get_filename_component(OPENCV_ROOT_DIR ${OPENCV_LIB_DIR} DIRECTORY)
        set(OPENCV_INCLUDE_DIR ${OPENCV_ROOT_DIR}/include/opencv4)
    endif()
endforeach()

if(OPENCV_LIBS)
    message("OpenCV libraries found")
    set(OPENCV_LIBS_FOUND TRUE)
else()
    set(OPENCV_ROOT_DIR ${DEPENDENCIES_DIR}/opencv)
    set(OPENCV_DEPENDENCIES_ARGS)
    set(OPENCV_EXTRA_LINKER_ARGS)
    set(OPENCV_PKGCONFIG)

    if(CMAKE_CROSSCOMPILING)
        set(FFMPEG_ROOT_DIR ${DEPENDENCIES_DIR}/ffmpeg)
        set(LIBX264_ROOT_DIR ${DEPENDENCIES_DIR}/x264)

        if (CMAKE_BUILD_TYPE STREQUAL Debug)
            set(CONFIGURE_DEBUG --enable-debug)
            set(OPENCV_DEBUG "-DBUILD_WITH_DEBUG_INFO=ON")
        endif()


        ExternalProject_Add(${X264_LIB}
            URL "https://code.videolan.org/videolan/x264/-/archive/${LIBX264_VERSION}/x264-${LIBX264_VERSION}.tar.gz"
            DOWNLOAD_DIR ${LIBX264_ROOT_DIR}
            PREFIX ${LIBX264_ROOT_DIR}
            CONFIGURE_COMMAND <SOURCE_DIR>/configure
            --host=${GNU_MACHINE}
            --enable-static
            --enable-shared
            --cross-prefix=${CROSS_PREFIX}
            --prefix=${CMAKE_BINARY_DIR}
            --extra-ldflags=-static-libstdc++
            --extra-cflags=-fPIC
            ${CONFIGURE_DEBUG}
            INSTALL_DIR ${CMAKE_BINARY_DIR}
            BUILD_COMMAND $(MAKE)
            INSTALL_COMMAND $(MAKE) install
            )

        set(FFMPEG_Config
            --enable-shared
            --enable-cross-compile
            --cross-prefix=${CROSS_PREFIX}
            --arch=${CMAKE_SYSTEM_PROCESSOR}
            --target-os=linux
            --prefix=${CMAKE_BINARY_DIR}
            --enable-gpl
            --enable-nonfree
            --enable-libx264
            --extra-cflags=-I${CMAKE_BINARY_DIR}/include
            --extra-cflags=-fPIC
            --extra-ldflags=-L${CMAKE_BINARY_DIR}/lib
            --extra-libs=-ldl
            --extra-libs=-static-libstdc++
        )

        ExternalProject_Add(${FFMPEG_LIB}
            URL "https://github.com/FFmpeg/FFmpeg/archive/n${FFMPEG_VERSION}.tar.gz"
            URL_HASH MD5=05792c611d1e3ebdf2c7003ff4467390
            DOWNLOAD_DIR ${FFMPEG_ROOT_DIR}
            PREFIX ${FFMPEG_ROOT_DIR}
            CONFIGURE_COMMAND <SOURCE_DIR>/configure ${FFMPEG_Config} ${CONFIGURE_DEBUG}
            INSTALL_DIR ${CMAKE_BINARY_DIR}
            BUILD_COMMAND $(MAKE) VERBOSE=1
            INSTALL_COMMAND $(MAKE) install
        )

        set(OPENCV_DEPENDENCIES_ARGS "-static-libstdc++ -Wl,-rpath,${CMAKE_BINARY_DIR}/lib")
        set(OPENCV_EXTRA_LINKER_ARGS "-DOPENCV_EXTRA_EXE_LINKER_FLAGS=${OPENCV_DEPENDENCIES_ARGS}")

        set(OPENCV_PKGCONFIG "PKG_CONFIG_LIBDIR=${CMAKE_BINARY_DIR}/lib/pkgconfig")

        set(FFMPEG_NAMES
            libavcodec.so
            libavformat.so
            libavutil.so
            libswscale.so
            )

        foreach(ffmpeg_lib ${FFMPEG_NAMES})
            add_library(FFMPEG_${ffmpeg_lib} SHARED IMPORTED)
            set_target_properties(FFMPEG_${ffmpeg_lib} PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/lib/${ffmpeg_lib})
            list(APPEND OPENCV_LIBS FFMPEG_${ffmpeg_lib})
        endforeach()

        add_library(X264_lib264.so SHARED IMPORTED)
        set_target_properties(X264_lib264.so PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/lib/libx264.so)
        list(APPEND OPENCV_LIBS X264_lib264.so)
    endif()

    set(OPENCV_CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_FLAGS=-fPIC
        -DCMAKE_CXX_FLAGS=-fPIC
        -DWITH_GTK=OFF
        -DWITH_JPEG=ON
        -DWITH_IPP=OFF
        -DBUILD_opencv_java_bindings_generator=OFF
        -DBUILD_opencv_ml=OFF
        -DBUILD_opencv_objdetect=OFF
        -DBUILD_opencv_photo=OFF
        -DBUILD_opencv_python_bindings_generator=OFF
        -DBUILD_opencv_stitching=OFF
        -DBUILD_opencv_gapi=OFF
        -DBUILD_opencv_features2d=OFF
        -DBUILD_opencv_dnn=OFF
        -DBUILD_opencv_flann=OFF
        -DBUILD_opencv_calib3d=OFF
        -DBUILD_opencv_python2=OFF
        -DBUILD_opencv_python3=OFF
        -DBUILD_opencv_java=OFF
        -DBUILD_opencv_js=OFF
        -DBUILD_opencv_ts=OFF
        -DBUILD_JPEG=ON
        -DBUILD_JPEG_TURBO_DISABLE=ON
        -DBUILD_PNG=ON
        -DBUILD_TIFF=ON
        -DZLIB_FOUND=OFF
        -DBUILD_ZLIB=ON
        -DBUILD_PERF_TESTS=OFF
        -DBUILD_TESTS=OFF
        -DBUILD_DOCS=OFF
        -DBUILD_opencv_apps=OFF
        -DBUILD_EXAMPLES=OFF
        -DWITH_V4L=ON
        -DWITH_LIBV4L=OFF
        -DWITH_FFMPEG=ON
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_INSTALL_RPATH=\$ORIGIN:\$ORIGIN/lib:\$ORIGIN/../lib
        -DCMAKE_SHARED_LINKER_FLAGS=-static-libstdc++
        ${OPENCV_DEBUG}
        )

    ExternalProject_Add(${OPENCV_LIB}
        URL "https://codeload.github.com/opencv/opencv/tar.gz/${OPENCV_VERSION}"
        URL_HASH MD5=f051c1ff7b327b60123d71b53801b316
        DOWNLOAD_DIR ${OPENCV_ROOT_DIR}
        PREFIX ${OPENCV_ROOT_DIR}
        CONFIGURE_COMMAND ${OPENCV_PKGCONFIG}
        ${CMAKE_COMMAND} ${OPENCV_CMAKE_ARGS} ${OPENCV_EXTRA_ARGS}
        ${OPENCV_EXTRA_LINKER_ARGS} ${OPENCV_ROOT_DIR}/src/${OPENCV_LIB}
        INSTALL_DIR ${CMAKE_BINARY_DIR}
        BUILD_COMMAND $(MAKE)
        INSTALL_COMMAND $(MAKE) install
        )

    if(CMAKE_CROSSCOMPILING)
        ExternalProject_Add_StepDependencies(${FFMPEG_LIB} build ${X264_LIB})
        ExternalProject_Add_StepDependencies(${OPENCV_LIB} build ${FFMPEG_LIB})
    endif()

    set(OPENCV_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include/opencv4)
    set(OPENCV_LIB_DIR ${CMAKE_BINARY_DIR}/lib)

    foreach(opencv_lib ${OPENCV_NAMES})
        add_library(OPENCV_${opencv_lib} SHARED IMPORTED)
        set_target_properties(OPENCV_${opencv_lib} PROPERTIES IMPORTED_LOCATION ${OPENCV_LIB_DIR}/${opencv_lib})
        list(APPEND OPENCV_LIBS OPENCV_${opencv_lib})
    endforeach()

endif()