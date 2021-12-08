# The Arm NN TensorFlow Lite delegate JNI (Experimental)

NOTE: This library is an experimental feature. We cannot guarentee full support for this.

'armnnDelegateJNI' is a library for accelerating certain TensorFlow Lite operators on Arm hardware specifically through Android
applications. Each release is packaged in an AAR which can be found on Maven Central.
The pre-built library contains the ArmNN Core, ArmNN Utils, Neon backend, CL Backend, and the ArmNN Delegate.
It is essential to only build these. The backends you choose are optional.

It requires a static build which can be switched on through setting BUILD_SHARED_LIBS=OFF. You will also have to set
CMAKE_ANDROID_STL_TYPE=c++_static when building ArmNN.

BUILD_DELEGATE_JNI_INTERFACE will also have to be set to true.

To download the prebuilt ArmNN Delegate JNI AAR from Maven Central, please go to [ArmNN Maven Central Release Page](https://search.maven.org/artifact/io.github.arm-software/armnn.delegate).
