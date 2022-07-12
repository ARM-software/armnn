# Arm NN Serialized Example model

This is a really simple example of an Arm NN serialized model which is supported on Netron. It is based on SSD MobileNet V1 FP32 model available in the public model zoo (https://github.com/ARM-software/ML-zoo/tree/master/models/object_detection/ssd_mobilenet_v1/tflite_fp32), converted using the ArmnnConverter tool. The model is trained by Google and uses 300x300 input image.
More details on how to deploy this model using Arm NN SDK can be found here: https://developer.arm.com/documentation/102274/latest.