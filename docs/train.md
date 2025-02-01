## Introduction

Now that we have a dataset of labeled samples from the active learning process, we can train a model on this dataset.


```python
import pandas as pd

df = pd.read_parquet("active_labeled.parquet")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filepath</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>data/imagenette/train/n02102040/n02102040_2788...</td>
      <td>English springer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>data/imagenette/train/n02102040/n02102040_3759...</td>
      <td>English springer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>data/imagenette/train/n02102040/n02102040_1916...</td>
      <td>English springer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>data/imagenette/train/n02102040/n02102040_6147...</td>
      <td>English springer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>data/imagenette/train/n02102040/n02102040_403....</td>
      <td>English springer</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>238</th>
      <td>data/imagenette/train/n03417042/n03417042_1869...</td>
      <td>garbage truck</td>
    </tr>
    <tr>
      <th>239</th>
      <td>data/imagenette/train/n02102040/n02102040_6763...</td>
      <td>English springer</td>
    </tr>
    <tr>
      <th>240</th>
      <td>data/imagenette/train/n01440764/n01440764_1455...</td>
      <td>tench</td>
    </tr>
    <tr>
      <th>241</th>
      <td>data/imagenette/train/n03028079/n03028079_2489...</td>
      <td>church</td>
    </tr>
    <tr>
      <th>242</th>
      <td>data/imagenette/train/n03425413/n03425413_2110...</td>
      <td>gas pump</td>
    </tr>
  </tbody>
</table>
<p>243 rows × 2 columns</p>
</div>



## Loading the data

We will use the fastai to train a model on this dataset. Feel free to use any other library you prefer.


```python
from fastai.vision.all import *

base_path = "."
dls = ImageDataLoaders.from_df(
    df,
    path=base_path,
    valid_pct=0.2,
    fn_col="filepath",
    label_col="label",
    bs=16,
    item_tfms=Resize(224),
    # batch_tfms=aug_transforms(size=224),
)

dls.show_batch()
```


    
![png](train_files/train_3_0.png)
    



```python
learn = vision_learner(dls, "vit_small_patch16_224", metrics=accuracy).to_fp16()
# learn.lr_find(suggest_funcs=(valley, slide))
```


```python
learn.fine_tune(10, base_lr=5e-3, freeze_epochs=3, cbs=ShowGraphCallback())

```

    /Users/dnth/Desktop/active-vision/.venv/lib/python3.12/site-packages/fastai/callback/fp16.py:47: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      self.autocast,self.learn.scaler,self.scales = autocast(dtype=dtype),GradScaler(**self.kwargs),L()
    /Users/dnth/Desktop/active-vision/.venv/lib/python3.12/site-packages/torch/amp/autocast_mode.py:266: UserWarning: User provided device_type of 'cuda', but CUDA is not available. Disabling
      warnings.warn(
    /Users/dnth/Desktop/active-vision/.venv/lib/python3.12/site-packages/fastai/callback/fp16.py:47: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      self.autocast,self.learn.scaler,self.scales = autocast(dtype=dtype),GradScaler(**self.kwargs),L()
    /Users/dnth/Desktop/active-vision/.venv/lib/python3.12/site-packages/torch/amp/grad_scaler.py:132: UserWarning: torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.
      warnings.warn(




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.229721</td>
      <td>1.472226</td>
      <td>0.541667</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.934528</td>
      <td>0.310813</td>
      <td>0.916667</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.189405</td>
      <td>0.198128</td>
      <td>0.958333</td>
      <td>00:02</td>
    </tr>
  </tbody>
</table>



    
![png](train_files/train_5_3.png)
    




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.050988</td>
      <td>0.175159</td>
      <td>0.958333</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.060211</td>
      <td>0.165249</td>
      <td>0.958333</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.039981</td>
      <td>0.193097</td>
      <td>0.937500</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.028811</td>
      <td>0.185286</td>
      <td>0.916667</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.021800</td>
      <td>0.170266</td>
      <td>0.916667</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.016554</td>
      <td>0.166251</td>
      <td>0.937500</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.017647</td>
      <td>0.179208</td>
      <td>0.916667</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.016122</td>
      <td>0.185861</td>
      <td>0.916667</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.013686</td>
      <td>0.192955</td>
      <td>0.916667</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.011061</td>
      <td>0.198078</td>
      <td>0.916667</td>
      <td>00:03</td>
    </tr>
  </tbody>
</table>



    
![png](train_files/train_5_6.png)
    


## Evaluating the model


```python
test_df = pd.read_parquet("evaluation_samples.parquet")
test_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filepath</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>data/imagenette/val/n03394916/n03394916_32422.JPEG</td>
      <td>French horn</td>
    </tr>
    <tr>
      <th>1</th>
      <td>data/imagenette/val/n03394916/n03394916_69132.JPEG</td>
      <td>French horn</td>
    </tr>
    <tr>
      <th>2</th>
      <td>data/imagenette/val/n03394916/n03394916_33771.JPEG</td>
      <td>French horn</td>
    </tr>
    <tr>
      <th>3</th>
      <td>data/imagenette/val/n03394916/n03394916_29940.JPEG</td>
      <td>French horn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>data/imagenette/val/n03394916/ILSVRC2012_val_00033682.JPEG</td>
      <td>French horn</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3920</th>
      <td>data/imagenette/val/n02979186/n02979186_27392.JPEG</td>
      <td>cassette player</td>
    </tr>
    <tr>
      <th>3921</th>
      <td>data/imagenette/val/n02979186/n02979186_2742.JPEG</td>
      <td>cassette player</td>
    </tr>
    <tr>
      <th>3922</th>
      <td>data/imagenette/val/n02979186/n02979186_2312.JPEG</td>
      <td>cassette player</td>
    </tr>
    <tr>
      <th>3923</th>
      <td>data/imagenette/val/n02979186/n02979186_12822.JPEG</td>
      <td>cassette player</td>
    </tr>
    <tr>
      <th>3924</th>
      <td>data/imagenette/val/n02979186/ILSVRC2012_val_00042982.JPEG</td>
      <td>cassette player</td>
    </tr>
  </tbody>
</table>
<p>3925 rows × 2 columns</p>
</div>




```python
filepaths = test_df["filepath"].tolist()
labels = test_df["label"].tolist()
test_dl = dls.test_dl(filepaths, bs=16)
preds, _, cls_preds = learn.get_preds(dl=test_dl, with_decoded=True)

results = pd.DataFrame(
    {
        "filepath": filepaths,
        "label": labels,
        "pred_label": [learn.dls.vocab[i] for i in cls_preds.numpy()],
    }
)

accuracy = float((results["label"] == results["pred_label"]).mean())
accuracy

```

    /Users/dnth/Desktop/active-vision/.venv/lib/python3.12/site-packages/fastai/callback/fp16.py:47: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      self.autocast,self.learn.scaler,self.scales = autocast(dtype=dtype),GradScaler(**self.kwargs),L()
    /Users/dnth/Desktop/active-vision/.venv/lib/python3.12/site-packages/torch/amp/autocast_mode.py:266: UserWarning: User provided device_type of 'cuda', but CUDA is not available. Disabling
      warnings.warn(
    /Users/dnth/Desktop/active-vision/.venv/lib/python3.12/site-packages/fastai/callback/fp16.py:47: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      self.autocast,self.learn.scaler,self.scales = autocast(dtype=dtype),GradScaler(**self.kwargs),L()
    /Users/dnth/Desktop/active-vision/.venv/lib/python3.12/site-packages/torch/amp/grad_scaler.py:132: UserWarning: torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.
      warnings.warn(




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>










    0.9936305732484076



With a mere 243 labeled samples, we have achieved an accuracy of 99.36% on the test set. The entire dataset contains over 9000 images, but it turns out that using active learning, we can achieve a high accuracy with a small number of labeled samples.
