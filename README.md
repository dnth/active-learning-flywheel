![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge)
[![PyPI](https://img.shields.io/pypi/v/active-vision?style=for-the-badge)](https://pypi.org/project/active-vision/)
![Downloads](https://img.shields.io/pepy/dt/active-vision?style=for-the-badge&logo=pypi&logoColor=white&label=Downloads&color=purple)   

<p align="center">
  <img src="https://raw.githubusercontent.com/dnth/active-vision/main/assets/logo.png" alt="active-vision">
</p>

Active learning at the edge for computer vision.

The goal of this project is to create a framework for the active learning loop for computer vision deployed on edge devices. 

Supported tasks:
- [X] Image classification
- [ ] Object detection
- [ ] Segmentation


## Installation

Get a release from PyPI
```bash
pip install active-vision
```

Install from source
```bash
git clone https://github.com/dnth/active-vision.git
cd active-vision
pip install -e .
```

I recommend using [uv](https://docs.astral.sh/uv/) to set up a virtual environment and install the package. You can also use other virtual env of your choice.

If you're using uv:

```bash
uv venv
uv sync
```
Once the virtual environment is created, you can install the package using pip.

> [!TIP]
> If you're using uv add a uv before the pip install command to install into your virtual environment. Eg:
> ```bash
> uv pip install active-vision
> ```

## Usage
See the [notebook](./nbs/04_relabel_loop.ipynb) for a complete example.

Be sure to prepared 3 datasets:
- [initial_samples](./nbs/initial_samples.parquet): A dataframe of an existing labeled training dataset to seed the training set.
- [unlabeled](./nbs/unlabeled_samples.parquet): A dataframe of unlabeled data which we will sample from using active learning.
- [eval](./nbs/evaluation_samples.parquet): A dataframe of labeled data which we will use to evaluate the performance of the model.

As a toy example I created the above 3 datasets from the imagenette dataset.

```python
from active_vision import ActiveLearner
import pandas as pd

# Create an active learner instance with a model
al = ActiveLearner("resnet18")

# Load dataset 
train_df = pd.read_parquet("training_samples.parquet")
al.load_dataset(df, filepath_col="filepath", label_col="label")

# Train model
al.train(epochs=3, lr=1e-3)

# Evaluate the model on a *labeled* evaluation set
accuracy = al.evaluate(eval_df, filepath_col="filepath", label_col="label")

# Get predictions from an *unlabeled* set
pred_df = al.predict(filepaths)

# Sample low confidence predictions from unlabeled set
uncertain_df = al.sample_uncertain(pred_df, num_samples=10)

# Launch a Gradio UI to label the low confidence samples
al.label(uncertain_df, output_filename="uncertain")
```

![Gradio UI](./assets/labeling_ui.png)

Once complete, the labeled samples will be save into a new df.
We can now add the newly labeled data to the training set.

```python
# Add newly labeled data to training set and save as a new file active_labeled
al.add_to_train_set(labeled_df, output_filename="active_labeled")
```

Repeat the process until the model is good enough. Use the dataset to train a larger model and deploy.

> [!TIP]
> For the toy dataset, I got to about 93% accuracy on the evaluation set with 200+ labeled images. The best performing model on the [leaderboard](https://github.com/fastai/imagenette) got 95.11% accuracy training on all 9469 labeled images.
> 
> This took me about 6 iterations of relabeling. Each iteration took about 5 minutes to complete including labeling and model training (resnet18). See the [notebook](./nbs/04_relabel_loop.ipynb) for more details.
>
> But using the dataset of 200+ images, I trained a more capable model (convnext_small_in22k) and got 99.3% accuracy on the evaluation set. See the [notebook](./nbs/05_retrain_larger.ipynb) for more details.

## Workflow
There are two workflows for active learning at the edge that we can use depending on the availability of labeled data.

### With unlabeled data
If we have no labeled data, we can use active learning to iteratively improve the model and build a labeled dataset.

1. Load a small proxy model.
2. Label an initial dataset. If there is none, you'll have to label some images.
3. Train the proxy model on the labeled dataset.
4. Run inference on the unlabeled dataset.
5. Evaluate the performance of the proxy model.
6. Is model good enough?    
    - Yes: Save the proxy model and the dataset.
    - No: Select the most informative images to label using active learning.
7. Label the most informative images and add them to the dataset.
8. Repeat steps 3-6.
9. Save the proxy model and the dataset.
10. Train a larger model on the saved dataset.



```mermaid
graph TD
    A[Load a small proxy model] --> B[Label an initial dataset]
    B --> C[Train proxy model on labeled dataset]
    C --> D[Run inference on unlabeled dataset]
    D --> E[Evaluate proxy model performance]
    E --> F{Model good enough?}
    F -->|Yes| G[Save proxy model and dataset]
    G --> H[Train and deploy a larger model]
    F -->|No| I[Select informative images using active learning]
    I --> J[Label selected images]
    J --> C
```
    
### With labeled data
If we have a labeled dataset, we can use active learning to iteratively improve the dataset and the model by fixing the most important label errors.

1. Load a small proxy model.
2. Train the proxy model on the labeled dataset.
3. Run inference on the entire labeled dataset.
4. Get the most important label errors with active learning.
5. Fix the label errors.
6. Repeat steps 2-5 until the dataset is good enough.
7. Save the labeled dataset.
8. Train a larger model on the saved labeled dataset.



```mermaid
graph TD
    A[Load a small proxy model] --> B[Train proxy model on labeled dataset]
    B --> C[Run inference on labeled dataset]
    C --> D[Get important label errors using active learning]
    D --> E[Fix label errors]
    E --> F{Dataset good enough?}
    F -->|No| B
    F -->|Yes| G[Save cleaned dataset]
    G --> H[Train and deploy larger model]
```


<!-- ## Methodology
To test out the workflows we will use the [imagenette dataset](https://huggingface.co/datasets/frgfm/imagenette). But this will be applicable to any dataset.

Imagenette is a subset of the ImageNet dataset with 10 classes. We will use this dataset to test out the workflows. Additionally, Imagenette has an existing leaderboard which we can use to evaluate the performance of the models.

### Step 1: Download the dataset
Download the imagenette dataset. The imagenette dataset has a train and validation split. Since the leaderboard is based on the validation set, we will evalutate the performance of our model on the validation set to make it easier to compare to the leaderboard.

We will treat the imagenette train set as a unlabeled set and iteratively sample from it while monitoring the performance on the validation set. Ideally we will be able to get to a point where the performance on the validation set is close to the leaderboard with minimal number of labeled images.

I've processed the imagenette dataset and uploaded it to the hub. You can download it from [here](https://huggingface.co/datasets/dnth/active-learning-imagenette).

To load the dataset, you can use the following code:
```python
from datasets import load_dataset

unlabeled_dataset = load_dataset("dnth/active-learning-imagenette", "unlabeled")
eval_dataset = load_dataset("dnth/active-learning-imagenette", "evaluation")
```

### Step 2: Initial Sampling
Label an initial dataset of 10 images from each class. This will give us a small proxy dataset to train our model on. The sampling will be done randomly. There are more intelligent sampling strategies but we will start with random sampling.

### Step 3: Training the proxy model
Train a proxy model on the initial dataset. The proxy model will be a small model that is easy to train and deploy. We will use the fastai framework to train the model. We will use the resnet18 architecture as a starting point. Once training is complete, compute the accuracy of the proxy model on the validation set and compare it to the leaderboard.

> [!TIP]
> With the initial model we got 91.24% accuracy on the validation set. See the [notebook](./nbs/01_initial_sampling.ipynb) for more details.
> | Train Epochs | Number of Images | Validation Accuracy |      Source      |
> |--------------|-----------------|----------------------|------------------|
> | 10           | 100             | 91.24%               | Initial sampling [notebook](./nbs/01_initial_sampling.ipynb) |
> | 80           | 9469            | 94.90%               | fastai |
> | 200          | 9469            | 95.11%               | fastai |



### Step 4: Inference on the unlabeled dataset
Run inference on the unlabeled dataset (the remaining imagenette train set) and evaluate the performance of the proxy model. 

### Step 5: Active learning
Use active learning to select the most informative images to label from the unlabeled set. Pick the top 10 images from the unlabeled set that the proxy model is least confident about and label them.

### Step 6: Repeat
Repeat step 3 - 5 until the performance on the validation set is close to the leaderboard. Note the number of labeled images vs the performance on the validation set. Ideally we want to get to a point where the performance on the validation set is close to the leaderboard with minimal number of labeled images.


After the first iteration we got 94.57% accuracy on the validation set. See the [notebook](./nbs/03_retrain_model.ipynb) for more details.

> [!TIP]
> | Train Epochs | Number of Images | Validation Accuracy |      Source      |
> |--------------|-----------------|----------------------|------------------|
> | 10           | 200             | 94.57%               | First relabeling [notebook](./nbs/03_retrain_model.ipynb) | -->
