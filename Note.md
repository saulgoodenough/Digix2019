## Huawei Age Prediction

### 特征提取





### Keras Multi GPU

From the Keras FAQs:

https://keras.io/getting-started/faq/#how-can-i-run-a-keras-model-on-multiple-gpus

Below is copy-pasted code to enable 'data parallelism'. I.e. having each of your GPUs process a different subset of your data independently.

```py
from keras.utils import multi_gpu_model

# Replicates `model` on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
parallel_model.fit(x, y, epochs=20, batch_size=256)
```

Note that this appears to be valid only for the Tensorflow backend at the time of writing.

**Update (Feb 2018)**:

Keras now accepts automatic gpu selection using multi_gpu_model, so you don't have to hardcode the number of gpus anymore. Details in this [Pull Request](https://github.com/keras-team/keras/pull/9226). In other words, this enables code that looks like this:

```py
try:
    model = multi_gpu_model(model)
except:
    pass
```

But to be more [explicit](https://www.python.org/dev/peps/pep-0020/), you can stick with something like:

```py
parallel_model = multi_gpu_model(model, gpus=None)
```

**Bonus**:

To check if you really are utilizing all of your GPUs, specifically NVIDIA ones, you can monitor your usage in the terminal using:

```py
watch -n0.5 nvidia-smi
```

References:

- https://keras.io/utils/#multi_gpu_model
- https://stackoverflow.com/questions/8223811/top-command-for-gpus-using-cuda