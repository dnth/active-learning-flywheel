## Download the dataset
In this example, we will use the Imagenette dataset to demonstrate how to prepare data for active learning.

First, lets download the dataset and extract it.


```python
!wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
!tar -xvzf imagenette2.tgz
!mv imagenette2 data/imagenette
!rm imagenette2.tgz
```

## Load the dataset

`active-vision` currently supports datasets in a pandas dataframe format. The dataframe should have at least 2 columns: `filepath` and `label`.


```python
from fastai.vision.all import get_image_files

path = "data/imagenette/train"
image_files = get_image_files(path)
len(image_files)
```




    9469




```python
lbl_dict = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

```


```python
import pandas as pd

# Create a dataframe with the filepath and label from parent directory
labels = [str(path.parts[-2]) for path in image_files]

# Map the labels to the label dictionary
labels = [lbl_dict[lbl] for lbl in labels]

df = pd.DataFrame({"filepath": [str(path) for path in image_files], "label": labels})

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
      <td>data/imagenette/train/n03394916/ILSVRC2012_val_00046669.JPEG</td>
      <td>French horn</td>
    </tr>
    <tr>
      <th>1</th>
      <td>data/imagenette/train/n03394916/n03394916_58454.JPEG</td>
      <td>French horn</td>
    </tr>
    <tr>
      <th>2</th>
      <td>data/imagenette/train/n03394916/n03394916_32588.JPEG</td>
      <td>French horn</td>
    </tr>
    <tr>
      <th>3</th>
      <td>data/imagenette/train/n03394916/n03394916_33663.JPEG</td>
      <td>French horn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>data/imagenette/train/n03394916/n03394916_27948.JPEG</td>
      <td>French horn</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9464</th>
      <td>data/imagenette/train/n02979186/n02979186_8089.JPEG</td>
      <td>cassette player</td>
    </tr>
    <tr>
      <th>9465</th>
      <td>data/imagenette/train/n02979186/n02979186_19444.JPEG</td>
      <td>cassette player</td>
    </tr>
    <tr>
      <th>9466</th>
      <td>data/imagenette/train/n02979186/n02979186_11074.JPEG</td>
      <td>cassette player</td>
    </tr>
    <tr>
      <th>9467</th>
      <td>data/imagenette/train/n02979186/n02979186_2938.JPEG</td>
      <td>cassette player</td>
    </tr>
    <tr>
      <th>9468</th>
      <td>data/imagenette/train/n02979186/n02979186_93.JPEG</td>
      <td>cassette player</td>
    </tr>
  </tbody>
</table>
<p>9469 rows × 2 columns</p>
</div>



## Initial samples
As an initial step, we will randomly sample 10 samples from each class. We will use these samples to kickstart the active learning process.


```python
initial_samples = (
    df.groupby("label")
    .apply(lambda x: x.sample(n=10, random_state=316))
    .reset_index(drop=True)
)

initial_samples
```

    /var/folders/9y/5mpk58851fq38f8ljx2svvnm0000gn/T/ipykernel_22109/414664958.py:3: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
      .apply(lambda x: x.sample(n=10, random_state=316))





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
      <td>data/imagenette/train/n02102040/n02102040_2788.JPEG</td>
      <td>English springer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>data/imagenette/train/n02102040/n02102040_3759.JPEG</td>
      <td>English springer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>data/imagenette/train/n02102040/n02102040_1916.JPEG</td>
      <td>English springer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>data/imagenette/train/n02102040/n02102040_6147.JPEG</td>
      <td>English springer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>data/imagenette/train/n02102040/n02102040_403.JPEG</td>
      <td>English springer</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>data/imagenette/train/n01440764/n01440764_10043.JPEG</td>
      <td>tench</td>
    </tr>
    <tr>
      <th>96</th>
      <td>data/imagenette/train/n01440764/n01440764_31535.JPEG</td>
      <td>tench</td>
    </tr>
    <tr>
      <th>97</th>
      <td>data/imagenette/train/n01440764/n01440764_12848.JPEG</td>
      <td>tench</td>
    </tr>
    <tr>
      <th>98</th>
      <td>data/imagenette/train/n01440764/n01440764_3997.JPEG</td>
      <td>tench</td>
    </tr>
    <tr>
      <th>99</th>
      <td>data/imagenette/train/n01440764/n01440764_29788.JPEG</td>
      <td>tench</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 2 columns</p>
</div>



Let's check the distribution of the labels.


```python
initial_samples["label"].value_counts()
```




    label
    English springer    10
    French horn         10
    cassette player     10
    chain saw           10
    church              10
    garbage truck       10
    gas pump            10
    golf ball           10
    parachute           10
    tench               10
    Name: count, dtype: int64



And save it to a parquet file.


```python
initial_samples.to_parquet("initial_samples.parquet")
```

## Unlabeled samples
For the remaining samples, we will use them as unlabeled samples. We will sample from these samples using active learning strategies.



```python
# Get the remaining samples by using pd.Index.difference
remaining_samples = df[~df.index.isin(initial_samples.index)].reset_index(drop=True)
remaining_samples

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
      <td>data/imagenette/train/n03394916/n03394916_4437.JPEG</td>
      <td>French horn</td>
    </tr>
    <tr>
      <th>1</th>
      <td>data/imagenette/train/n03394916/n03394916_42413.JPEG</td>
      <td>French horn</td>
    </tr>
    <tr>
      <th>2</th>
      <td>data/imagenette/train/n03394916/n03394916_38808.JPEG</td>
      <td>French horn</td>
    </tr>
    <tr>
      <th>3</th>
      <td>data/imagenette/train/n03394916/n03394916_24128.JPEG</td>
      <td>French horn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>data/imagenette/train/n03394916/n03394916_11289.JPEG</td>
      <td>French horn</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9364</th>
      <td>data/imagenette/train/n02979186/n02979186_8089.JPEG</td>
      <td>cassette player</td>
    </tr>
    <tr>
      <th>9365</th>
      <td>data/imagenette/train/n02979186/n02979186_19444.JPEG</td>
      <td>cassette player</td>
    </tr>
    <tr>
      <th>9366</th>
      <td>data/imagenette/train/n02979186/n02979186_11074.JPEG</td>
      <td>cassette player</td>
    </tr>
    <tr>
      <th>9367</th>
      <td>data/imagenette/train/n02979186/n02979186_2938.JPEG</td>
      <td>cassette player</td>
    </tr>
    <tr>
      <th>9368</th>
      <td>data/imagenette/train/n02979186/n02979186_93.JPEG</td>
      <td>cassette player</td>
    </tr>
  </tbody>
</table>
<p>9369 rows × 2 columns</p>
</div>




```python
remaining_samples.to_parquet("unlabeled_samples.parquet")
```

## Evaluation samples

Now let's create the evaluation samples which will be used to evaluate the performance of the model. We will use the validation set from the Imagenette dataset as the evaluation set.




```python
path = "data/imagenette/val"
image_files = get_image_files(path)
len(image_files)

```




    3925




```python
labels = [str(path.parts[-2]) for path in image_files]

# Map the labels to the label dictionary
labels = [lbl_dict[lbl] for lbl in labels]

evaluation_samples = pd.DataFrame(
    {"filepath": [str(path) for path in image_files], "label": labels}
)

evaluation_samples
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
evaluation_samples.to_parquet("evaluation_samples.parquet")
```

We are now ready to start the active learning process.
