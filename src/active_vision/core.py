import pandas as pd
from loguru import logger
from fastai.vision.models import resnet18, resnet34
from fastai.callback.all import ShowGraphCallback
from fastai.vision.all import (
    ImageDataLoaders,
    aug_transforms,
    Resize,
    vision_learner,
    accuracy,
    valley,
    slide,
    minimum,
    steep,
)
import torch
import torch.nn.functional as F

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class ActiveLearner:
    def __init__(self, model_name: str):
        self.model = self.load_model(model_name)

    def load_model(self, model_name: str):
        models = {"resnet18": resnet18, "resnet34": resnet34}
        logger.info(f"Loading model {model_name}")
        if model_name not in models:
            logger.error(f"Model {model_name} not found")
            raise ValueError(f"Model {model_name} not found")
        return models[model_name]

    def load_dataset(
        self,
        df: pd.DataFrame,
        filepath_col: str,
        label_col: str,
        valid_pct: float = 0.2,
        batch_size: int = 16,
        image_size: int = 224,
    ):
        logger.info(f"Loading dataset from {filepath_col} and {label_col}")
        self.train_set = df.copy()

        logger.info("Creating dataloaders")
        self.dls = ImageDataLoaders.from_df(
            df,
            path=".",
            valid_pct=valid_pct,
            fn_col=filepath_col,
            label_col=label_col,
            bs=batch_size,
            item_tfms=Resize(image_size),
            batch_tfms=aug_transforms(size=image_size, min_scale=0.75),
        )
        logger.info("Creating learner")
        self.learn = vision_learner(self.dls, self.model, metrics=accuracy).to_fp16()
        self.class_names = self.dls.vocab
        logger.info("Done. Ready to train.")

    def lr_find(self):
        logger.info("Finding optimal learning rate")
        self.lrs = self.learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
        logger.info(f"Optimal learning rate: {self.lrs.valley}")

    def train(self, epochs: int, lr: float):
        logger.info(f"Training for {epochs} epochs with learning rate: {lr}")
        self.learn.fine_tune(epochs, lr, cbs=[ShowGraphCallback()])

    def predict(self, filepaths: list[str], batch_size: int = 16):
        """
        Run inference on an unlabeled dataset. Returns a df with filepaths and predicted labels, and confidence scores.
        """
        logger.info(f"Running inference on {len(filepaths)} samples")
        test_dl = self.dls.test_dl(filepaths, bs=batch_size)
        preds, _, cls_preds = self.learn.get_preds(dl=test_dl, with_decoded=True)

        self.pred_df = pd.DataFrame(
            {
                "filepath": filepaths,
                "pred_label": [self.learn.dls.vocab[i] for i in cls_preds.numpy()],
                "pred_conf": torch.max(F.softmax(preds, dim=1), dim=1)[0].numpy(),
            }
        )
        return self.pred_df

    def evaluate(self, df: pd.DataFrame, filepath_col: str, label_col: str, batch_size: int = 16):
        """
        Evaluate on a labeled dataset. Returns a score.
        """
        self.eval_set = df.copy()

        filepaths = self.eval_set[filepath_col].tolist()
        labels = self.eval_set[label_col].tolist()
        test_dl = self.dls.test_dl(filepaths, bs=batch_size)
        preds, _, cls_preds = self.learn.get_preds(dl=test_dl, with_decoded=True)

        self.eval_df = pd.DataFrame(
            {
                "filepath": filepaths,
                "label": labels,
                "pred_label": [self.learn.dls.vocab[i] for i in cls_preds.numpy()],
            }
        )

        accuracy = float((self.eval_df["label"] == self.eval_df["pred_label"]).mean())
        logger.info(f"Accuracy: {accuracy:.2%}")
        return accuracy

    def sample_uncertain(self, num_samples: int):
        """
        Sample top `num_samples` low confidence samples. Returns a df with filepaths and predicted labels, and confidence scores.
        """
        uncertain_df = self.pred_df.sort_values(
            by="pred_conf", ascending=True
        ).head(num_samples)
        return uncertain_df

    def add_to_train_set(self, df: pd.DataFrame):
        """
        Add samples to the training set.
        """
        new_train_set = df.copy()
        new_train_set.drop(columns=["pred_conf"], inplace=True)
        new_train_set.rename(columns={"pred_label": "label"}, inplace=True)

        len_old = len(self.train_set)

        logger.info(f"Adding {len(new_train_set)} samples to training set")
        self.train_set = pd.concat([self.train_set, new_train_set])

        self.train_set = self.train_set.drop_duplicates(
            subset=["filepath"], keep="last"
        )
        self.train_set.reset_index(drop=True, inplace=True)


        if len(self.train_set) == len_old:
            logger.warning("No new samples added to training set")

        elif len_old + len(new_train_set) < len(self.train_set):
            logger.warning("Some samples were duplicates and removed from training set")

        else:
            logger.info("All new samples added to training set")
            logger.info(f"Training set now has {len(self.train_set)} samples")
