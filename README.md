# active-learning-flywheel
Active learning at the edge for computer vision.

The goal of this project is to create a framework for active learning at the edge for computer vision. We should be able to train a model on a small dataset and then use active learning to iteratively improve the model all on a local machine.

## Tech Stack

- Training framework: fastai
- User interface: streamlit
- Database: sqlite
- Experiment tracking: wandb

## Workflow

1. Load a small (proxy) model.
2. Label an initial dataset.
3. Train the model on the labeled dataset.
4. Run inference on the unlabeled dataset.
5. Evaluate the performance of the model on the unlabeled dataset.
6. Is model good enough?    
    - Yes: Save the model and the dataset.
    - No: Select the most informative images to label using active learning.
7. Label the most informative images and add them to the dataset.
8. Repeat steps 3-6.
9. Save the model and the dataset.
10. Train a larger model on the saved dataset.



```mermaid
graph TD
    A[Load a small (proxy) model] --> B[Label an initial dataset]
    B --> C[Train model on labeled dataset]
    C --> D[Run inference on unlabeled dataset]
    D --> E[Evaluate model performance]
    E --> F{Model good enough?}
    F -->|Yes| G[Save model and dataset]
    G --> H[Train a larger model than the proxy model]
    F -->|No| I[Select informative images]
    I --> J[Label selected images]
    J --> C
```
    