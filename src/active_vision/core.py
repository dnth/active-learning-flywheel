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

    def evaluate(
        self, df: pd.DataFrame, filepath_col: str, label_col: str, batch_size: int = 16
    ):
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

    def sample_uncertain(self, df: pd.DataFrame, num_samples: int):
        """
        Sample top `num_samples` low confidence samples. Returns a df with filepaths and predicted labels, and confidence scores.
        """
        logger.info(f"Getting top {num_samples} low confidence samples")
        uncertain_df = df.sort_values(by="pred_conf", ascending=True).head(num_samples)
        return uncertain_df

    def label(self, df: pd.DataFrame, output_filename: str = "labeled"):
        """
        Launch a labeling interface for the user to label the samples.
        Input is a df with filepaths listing the files to be labeled. Output is a df with filepaths and labels.
        """
        import gradio as gr

        shortcut_js = """
        <script>
        function shortcuts(e) {
            // Only block shortcuts if we're in a text input or textarea
            if (e.target.tagName.toLowerCase() === "textarea" || 
                (e.target.tagName.toLowerCase() === "input" && e.target.type.toLowerCase() === "text")) {
                return;
            }
            
            if (e.key.toLowerCase() == "w") {
                document.getElementById("submit_btn").click();
            } else if (e.key.toLowerCase() == "d") {
                document.getElementById("next_btn").click();
            } else if (e.key.toLowerCase() == "a") {
                document.getElementById("back_btn").click();
            }
        }
        document.addEventListener('keypress', shortcuts, false);
        </script>
        """

        logger.info(f"Launching labeling interface for {len(df)} samples")

        filepaths = df["filepath"].tolist()

        with gr.Blocks(head=shortcut_js) as demo:
            current_index = gr.State(value=0)

            filename = gr.Textbox(
                label="Filename", value=filepaths[0], interactive=False
            )

            image = gr.Image(
                type="filepath", label="Image", value=filepaths[0], height=500
            )
            category = gr.Radio(choices=self.class_names, label="Select Category")

            with gr.Row():
                back_btn = gr.Button("← Previous (A)", elem_id="back_btn")
                submit_btn = gr.Button(
                    "Submit (W)",
                    variant="primary",
                    elem_id="submit_btn",
                    interactive=False,
                )
                next_btn = gr.Button("Next → (D)", elem_id="next_btn")

            progress = gr.Slider(
                minimum=0,
                maximum=len(filepaths) - 1,
                value=0,
                label="Progress",
                interactive=False,
            )

            finish_btn = gr.Button("Finish Labeling", variant="primary")

            def update_submit_btn(choice):
                return gr.Button(interactive=choice is not None)

            category.change(
                fn=update_submit_btn, inputs=[category], outputs=[submit_btn]
            )

            def navigate(current_idx, direction):
                next_idx = current_idx + direction
                if 0 <= next_idx < len(filepaths):
                    return filepaths[next_idx], filepaths[next_idx], next_idx, next_idx
                return (
                    filepaths[current_idx],
                    filepaths[current_idx],
                    current_idx,
                    current_idx,
                )

            def save_and_next(current_idx, selected_category):
                if selected_category is None:
                    return (
                        filepaths[current_idx],
                        filepaths[current_idx],
                        current_idx,
                        current_idx,
                    )

                # Save the current annotation
                with open(f"{output_filename}.csv", "a") as f:
                    f.write(f"{filepaths[current_idx]},{selected_category}\n")

                # Move to next image if not at the end
                next_idx = current_idx + 1
                if next_idx >= len(filepaths):
                    return (
                        filepaths[current_idx],
                        filepaths[current_idx],
                        current_idx,
                        current_idx,
                    )
                return filepaths[next_idx], filepaths[next_idx], next_idx, next_idx

            def convert_csv_to_parquet():
                try:
                    df = pd.read_csv(f"{output_filename}.csv", header=None)
                    df.columns = ["filepath", "label"]
                    df = df.drop_duplicates(subset=["filepath"], keep="last")
                    df.to_parquet(f"{output_filename}.parquet")
                    gr.Info(f"Annotation saved to {output_filename}.parquet")
                except Exception as e:
                    logger.error(e)
                    return

            back_btn.click(
                fn=lambda idx: navigate(idx, -1),
                inputs=[current_index],
                outputs=[filename, image, current_index, progress],
            )

            next_btn.click(
                fn=lambda idx: navigate(idx, 1),
                inputs=[current_index],
                outputs=[filename, image, current_index, progress],
            )

            submit_btn.click(
                fn=save_and_next,
                inputs=[current_index, category],
                outputs=[filename, image, current_index, progress],
            )

            finish_btn.click(fn=convert_csv_to_parquet)

        demo.launch(height=1000)

    def add_to_train_set(self, df: pd.DataFrame, output_filename: str):
        """
        Add samples to the training set.
        """
        new_train_set = df.copy()
        # new_train_set.drop(columns=["pred_conf"], inplace=True)
        # new_train_set.rename(columns={"pred_label": "label"}, inplace=True)

        # len_old = len(self.train_set)

        logger.info(f"Adding {len(new_train_set)} samples to training set")
        self.train_set = pd.concat([self.train_set, new_train_set])

        self.train_set = self.train_set.drop_duplicates(
            subset=["filepath"], keep="last"
        )
        self.train_set.reset_index(drop=True, inplace=True)

        self.train_set.to_parquet(f"{output_filename}.parquet")
        logger.info(f"Saved training set to {output_filename}.parquet")

        # if len(self.train_set) == len_old:
        #     logger.warning("No new samples added to training set")

        # elif len_old + len(new_train_set) < len(self.train_set):
        #     logger.warning("Some samples were duplicates and removed from training set")

        # else:
        #     logger.info("All new samples added to training set")
        #     logger.info(f"Training set now has {len(self.train_set)} samples")
