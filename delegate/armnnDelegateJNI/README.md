# The Arm NN TensorFlow Lite delegate JNI (Experimental)

NOTE: This library is an experimental feature. We cannot guarentee full support for this.

It requires a static build which can be switched on through setting BUILD_SHARED_LIBS=OFF. You will also have to set
CMAKE_ANDROID_STL_TYPE=c++_static when building ArmNN.

BUILD_DELEGATE_JNI_INTERFACE will also have to be set to true.
