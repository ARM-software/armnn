# TfLite Delegate Quick Start Guide
If you have downloaded the Arm NN Github binaries or built the TfLite delegate yourself, then this tutorial will show you how you can
integrate it into TfLite to run models using python.

Here is an example python script showing how to do this. In this script we are making use of the 
[external adaptor](https://www.tensorflow.org/lite/performance/implementing_delegate#option_2_leverage_external_delegate) 
tool of TfLite that allows you to load delegates at runtime.
```python
import numpy as np
import tflite_runtime.interpreter as tflite

# Load TFLite model and allocate tensors.
# (if you are using the complete tensorflow package you can find load_delegate in tf.experimental.load_delegate)
armnn_delegate = tflite.load_delegate( library="<path-to-armnn-binaries>/libarmnnDelegate.so",
                                       options={"backends": "CpuAcc,GpuAcc,CpuRef", "logging-severity":"info"})
# Delegates/Executes all operations supported by Arm NN to/with Arm NN
interpreter = tflite.Interpreter(model_path="<your-armnn-repo-dir>/delegate/python/test/test_data/mock_model.tflite", 
                                 experimental_delegates=[armnn_delegate])
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# Print out result
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
```

# Prepare the environment
Pre-requisites:
 * Dynamically build Arm NN Delegate library or download the Arm NN binaries
 * python3 (Depends on TfLite version)
 * virtualenv
 * numpy (Depends on TfLite version)
 * tflite_runtime (>=2.5, depends on Arm NN Delegate)

If you haven't built the delegate yet then take a look at the [build guide](./BuildGuideNative.md). Otherwise, 
you can download the binaries [here](https://github.com/ARM-software/armnn/releases/)

We recommend creating a virtual environment for this tutorial. For the following code to work python3 is needed. Please
also check the documentation of the TfLite version you want to use. There might be additional prerequisites for the python
version. We will use Tensorflow Lite 2.5.0 for this guide.
```bash
# Install python3 (We ended up with python3.5.3) and virtualenv
sudo apt-get install python3-pip
sudo pip3 install virtualenv

# create a virtual environment
cd your/tutorial/dir
# creates a directory myenv at the current location
virtualenv -p python3 myenv 
# activate the environment
source myenv/bin/activate
```

Now that the environment is active we can install additional packages we need for our example script. As you can see 
in the python script at the start of this page, this tutorial uses the `tflite_runtime` rather than the whole tensorflow 
package. The `tflite_runtime` is a package that wraps the TfLite Interpreter. Therefore it can only be used to run inferences of 
TfLite models. But since Arm NN is only an inference engine itself this is a perfect match. The 
`tflite_runtime` is also much smaller than the whole tensorflow package and better suited to run models on 
mobile and embedded devices.

The TfLite [website](https://www.tensorflow.org/lite/guide/python) shows you two methods to download the `tflite_runtime`  package. 
In our experience, the use of the pip command works for most systems including debian. However, if you're using an older version of Tensorflow, 
you may need to build the pip package from source. You can find more information [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/pip_package/README.md).
But in our case, with Tensorflow Lite 2.5.0, we can install through:

```
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

Your virtual environment is now all setup. Copy the final python script into a python file e.g. 
`ExternalDelegatePythonTutorial.py`. Modify the python script above and replace `<path-to-armnn-binaries>` and 
`<your-armnn-repo-dir>` with the directories you have set up. If you've been using the [native build guide](./BuildGuideNative.md) 
this will be `$BASEDIR/armnn/build` and `$BASEDIR/armnn`.

Finally, execute the script:
```bash
python ExternalDelegatePythonTutorial.py
```
The output should look similar to this:
```bash
Info: Arm NN v28.0.0

Info: Initialization time: 0.56 ms

INFO: TfLiteArmnnDelegate: Created TfLite Arm NN delegate.
[[ 12 123  16  12  11  14  20  16  20  12]]
Info: Shutdown time: 0.28 ms
```

For more details of the kind of options you can pass to the Arm NN delegate please check the parameters of function tflite_plugin_create_delegate.

You can also test the functionality of the external delegate adaptor by running some unit tests:
```bash
pip install pytest
cd armnn/delegate/python/test
# You can deselect tests that require backends that your hardware doesn't support using markers e.g. -m "not GpuAccTest"
pytest --delegate-dir="<path-to-armnn-binaries>/libarmnnDelegate.so" -m "not GpuAccTest"
```
