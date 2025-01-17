import pandas as pd
from loguru import logger
from fastai.vision.all import *
import torch
import torch.nn.functional as F

import warnings
from typing import Callable

warnings.filterwarnings("ignore", category=FutureWarning)


class ActiveLearner:
    """
    Active Learning framework for computer vision tasks.

    Attributes:
        Model Related:
            model: The base model architecture (str or Callable)
            learn: fastai Learner object for training
            lrs: Learning rate finder results

        Data Related:
            train_set (pd.DataFrame): Training dataset
            eval_set (pd.DataFrame): Evaluation dataset with ground truth labels
            dls: fastai DataLoaders object
            class_names: List of class names from the dataset
            num_classes (int): Number of classes in the dataset

        Prediction Related:
            pred_df (pd.DataFrame): Predictions on a dataframe
            eval_df (pd.DataFrame): Predictions on evaluation data
    """

    def __init__(self, model_name: str | Callable):
        self.model = self.load_model(model_name)

    def load_model(self, model_name: str | Callable):
        if isinstance(model_name, Callable):
            logger.info(f"Loading fastai model {model_name.__name__}")
            return model_name

        if isinstance(model_name, str):
            logger.info(f"Loading timm model {model_name}")
            return model_name

    def load_dataset(
        self,
        df: pd.DataFrame,
        filepath_col: str,
        label_col: str,
        valid_pct: float = 0.2,
        batch_size: int = 16,
        image_size: int = 224,
        batch_tfms: Callable = None,
        learner_path: str = None,
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
            batch_tfms=batch_tfms,
        )

        if learner_path:
            logger.info(f"Loading learner from {learner_path}")
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                logger.info(f"Loading learner on GPU.")
            else:
                logger.info(f"Loading learner on CPU.")

            self.learn = load_learner(learner_path, cpu=not gpu_available)
        else:
            logger.info("Creating learner")
            self.learn = vision_learner(
                self.dls, self.model, metrics=accuracy
            ).to_fp16()

        self.class_names = self.dls.vocab
        self.num_classes = self.dls.c
        logger.info("Done. Ready to train.")

    def show_batch(
        self,
        num_samples: int = 9,
        unique: bool = False,
        num_rows: int = None,
        num_cols: int = None,
    ):
        """
        Show a batch of images from the dataset.

        Args:
            num_samples: Number of samples to show.
            unique: Whether to show unique samples.
            num_rows: Number of rows in the grid.
            num_cols: Number of columns in the grid.
        """
        self.dls.show_batch(
            max_n=num_samples, unique=unique, nrows=num_rows, ncols=num_cols
        )

    def lr_find(self):
        logger.info("Finding optimal learning rate")
        self.lrs = self.learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
        logger.info(f"Optimal learning rate: {self.lrs.valley}")

    def train(self, epochs: int, lr: float, head_tuning_epochs: int = 1):
        """
        Train the model.

        Args:
            epochs: Number of epochs to train for.
            lr: Learning rate.
            head_tuning_epochs: Number of epochs to train the head.
        """
        logger.info(f"Training head for {head_tuning_epochs} epochs")
        logger.info(f"Training model end-to-end for {epochs} epochs")
        logger.info(f"Learning rate: {lr} with one-cycle learning rate scheduler")
        self.learn.fine_tune(
            epochs, lr, freeze_epochs=head_tuning_epochs, cbs=[ShowGraphCallback()]
        )

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

    def sample_uncertain(
        self, df: pd.DataFrame, num_samples: int, strategy: str = "least-confidence"
    ):
        """
        Sample top `num_samples` low confidence samples. Returns a df with filepaths and predicted labels, and confidence scores.

        Strategies:
        - least-confidence: Get top `num_samples` low confidence samples.
        - margin-of-confidence: Get top `num_samples` samples with the smallest margin between the top two predictions.
        - ratio-of-confidence: Get top `num_samples` samples with the highest ratio between the top two predictions.
        - entropy: Get top `num_samples` samples with the highest entropy.
        """

        # Remove samples that is already in the training set
        df = df[~df["filepath"].isin(self.train_set["filepath"])].copy()

        if strategy == "least-confidence":
            logger.info(f"Getting top {num_samples} low confidence samples")

            df.loc[:, "uncertainty_score"] = 1 - (df["pred_conf"]) / (
                self.num_classes - (self.num_classes - 1)
            )

            # Sort by descending uncertainty score
            uncertain_df = df.sort_values(by="uncertainty_score", ascending=False).head(
                num_samples
            )
            return uncertain_df

        # TODO: Implement margin of confidence strategy
        elif strategy == "margin-of-confidence":
            logger.error("Margin of confidence strategy not implemented")
            raise NotImplementedError("Margin of confidence strategy not implemented")

        # TODO: Implement ratio of confidence strategy
        elif strategy == "ratio-of-confidence":
            logger.error("Ratio of confidence strategy not implemented")
            raise NotImplementedError("Ratio of confidence strategy not implemented")

        # TODO: Implement entropy strategy
        elif strategy == "entropy":
            logger.error("Entropy strategy not implemented")
            raise NotImplementedError("Entropy strategy not implemented")

        else:
            logger.error(f"Unknown strategy: {strategy}")
            raise ValueError(f"Unknown strategy: {strategy}")

    def sample_diverse(self, df: pd.DataFrame, num_samples: int):
        """
        Sample top `num_samples` diverse samples. Returns a df with filepaths and predicted labels, and confidence scores.

        Strategies:
        - model-based-outlier: Get top `num_samples` samples with lowest activation of the model's last layer.
        - cluster-based: Get top `num_samples` samples with the highest distance to the nearest neighbor.
        - representative: Get top `num_samples` samples with the highest distance to the centroid of the training set.
        """
        logger.error("Diverse sampling strategy not implemented")
        raise NotImplementedError("Diverse sampling strategy not implemented")

    def sample_random(self, df: pd.DataFrame, num_samples: int, seed: int = None):
        """
        Sample `num_samples` random samples. Returns a df with filepaths and predicted labels, and confidence scores.
        """

        logger.info(f"Sampling {num_samples} random samples")
        if seed is not None:
            logger.info(f"Using seed: {seed}")
        return df.sample(n=num_samples, random_state=seed)

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
            
            if (e.key === "ArrowUp" || e.key === "Enter") {
                document.getElementById("submit_btn").click();
            } else if (e.key === "ArrowRight") {
                document.getElementById("next_btn").click();
            } else if (e.key === "ArrowLeft") {
                document.getElementById("back_btn").click();
            }
        }
        document.addEventListener('keydown', shortcuts, false);
        </script>
        """

        logger.info(f"Launching labeling interface for {len(df)} samples")

        filepaths = df["filepath"].tolist()

        with gr.Blocks(head=shortcut_js) as demo:
            current_index = gr.State(value=0)

            image = gr.Image(
                type="filepath", label="Image", value=filepaths[0], height=500
            )

            with gr.Row():
                filename = gr.Textbox(
                    label="Filename", value=filepaths[0], interactive=False
                )

                pred_label = gr.Textbox(
                    label="Predicted Label",
                    value=df["pred_label"].iloc[0]
                    if "pred_label" in df.columns
                    else "",
                    interactive=False,
                )
                pred_conf = gr.Textbox(
                    label="Confidence",
                    value=f"{df['pred_conf'].iloc[0]:.2%}"
                    if "pred_conf" in df.columns
                    else "",
                    interactive=False,
                )

            category = gr.Radio(
                choices=self.class_names,
                label="Select Category",
                value=df["pred_label"].iloc[0] if "pred_label" in df.columns else None,
            )

            with gr.Row():
                back_btn = gr.Button("← Previous", elem_id="back_btn")
                submit_btn = gr.Button(
                    "Submit (↑/Enter)",
                    variant="primary",
                    elem_id="submit_btn",
                    interactive=False,
                )
                next_btn = gr.Button("Next →", elem_id="next_btn")

            progress = gr.Slider(
                minimum=0,
                maximum=len(filepaths) - 1,
                value=0,
                label="Progress",
                interactive=False,
            )

            finish_btn = gr.Button("Finish Labeling", variant="primary")

            with gr.Accordion("Zero-Shot Inference", open=False) as zero_shot_accordion:
                gr.Markdown("""
                Uses a VLM to predict the label of the image.
                """)

                import xinfer
                from xinfer.model_registry import model_registry
                from xinfer.types import ModelInputOutput

                # Get models and filter for image-to-text models
                all_models = model_registry.list_models()
                model_list = [
                    model.id
                    for model in all_models
                    if model.input_output == ModelInputOutput.IMAGE_TEXT_TO_TEXT
                ]

                with gr.Row():
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=model_list,
                            label="Select a model",
                            value="vikhyatk/moondream2",
                        )
                        device_dropdown = gr.Dropdown(
                            choices=["cuda", "cpu"],
                            label="Device",
                            value="cuda" if torch.cuda.is_available() else "cpu",
                        )
                        dtype_dropdown = gr.Dropdown(
                            choices=["float32", "float16", "bfloat16"],
                            label="Data Type",
                            value="float16" if torch.cuda.is_available() else "float32",
                        )

                with gr.Column():
                    prompt_textbox = gr.Textbox(
                        label="Prompt",
                        lines=3,
                        value=f"Classify the image into one of the following categories: {self.class_names}",
                        interactive=True,
                    )
                    inference_btn = gr.Button("Run Inference", variant="primary")

                    result_textbox = gr.Textbox(
                        label="Result",
                        lines=3,
                        interactive=False,
                    )

            def run_zero_shot_inference(prompt, model, device, dtype, current_filename):
                model = xinfer.create_model(model, device=device, dtype=dtype)
                result = model.infer(current_filename, prompt).text
                return result

            inference_btn.click(
                fn=run_zero_shot_inference,
                inputs=[
                    prompt_textbox,
                    model_dropdown,
                    device_dropdown,
                    dtype_dropdown,
                    filename,
                ],
                outputs=[result_textbox],
            )

            def update_submit_btn(choice):
                return gr.Button(interactive=choice is not None)

            category.change(
                fn=update_submit_btn, inputs=[category], outputs=[submit_btn]
            )

            def navigate(current_idx, direction):
                # Convert current_idx to int before arithmetic
                current_idx = int(current_idx)
                next_idx = current_idx + direction

                if 0 <= next_idx < len(filepaths):
                    return (
                        filepaths[next_idx],
                        filepaths[next_idx],
                        df["pred_label"].iloc[next_idx]
                        if "pred_label" in df.columns
                        else "",
                        f"{df['pred_conf'].iloc[next_idx]:.2%}"
                        if "pred_conf" in df.columns
                        else "",
                        df["pred_label"].iloc[next_idx]
                        if "pred_label" in df.columns
                        else None,
                        next_idx,
                        next_idx,
                    )
                return (
                    filepaths[current_idx],
                    filepaths[current_idx],
                    df["pred_label"].iloc[current_idx]
                    if "pred_label" in df.columns
                    else "",
                    f"{df['pred_conf'].iloc[current_idx]:.2%}"
                    if "pred_conf" in df.columns
                    else "",
                    df["pred_label"].iloc[current_idx]
                    if "pred_label" in df.columns
                    else None,
                    current_idx,
                    current_idx,
                )

            def save_and_next(current_idx, selected_category):
                # Convert current_idx to int before arithmetic
                current_idx = int(current_idx)

                if selected_category is None:
                    return (
                        filepaths[current_idx],
                        filepaths[current_idx],
                        df["pred_label"].iloc[current_idx]
                        if "pred_label" in df.columns
                        else "",
                        f"{df['pred_conf'].iloc[current_idx]:.2%}"
                        if "pred_conf" in df.columns
                        else "",
                        df["pred_label"].iloc[current_idx]
                        if "pred_label" in df.columns
                        else None,
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
                        df["pred_label"].iloc[current_idx]
                        if "pred_label" in df.columns
                        else "",
                        f"{df['pred_conf'].iloc[current_idx]:.2%}"
                        if "pred_conf" in df.columns
                        else "",
                        df["pred_label"].iloc[current_idx]
                        if "pred_label" in df.columns
                        else None,
                        current_idx,
                        current_idx,
                    )
                return (
                    filepaths[next_idx],
                    filepaths[next_idx],
                    df["pred_label"].iloc[next_idx]
                    if "pred_label" in df.columns
                    else "",
                    f"{df['pred_conf'].iloc[next_idx]:.2%}"
                    if "pred_conf" in df.columns
                    else "",
                    df["pred_label"].iloc[next_idx]
                    if "pred_label" in df.columns
                    else None,
                    next_idx,
                    next_idx,
                )

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
                outputs=[
                    filename,
                    image,
                    pred_label,
                    pred_conf,
                    category,
                    current_index,
                    progress,
                ],
            )

            next_btn.click(
                fn=lambda idx: navigate(idx, 1),
                inputs=[current_index],
                outputs=[
                    filename,
                    image,
                    pred_label,
                    pred_conf,
                    category,
                    current_index,
                    progress,
                ],
            )

            submit_btn.click(
                fn=save_and_next,
                inputs=[current_index, category],
                outputs=[
                    filename,
                    image,
                    pred_label,
                    pred_conf,
                    category,
                    current_index,
                    progress,
                ],
            )

            finish_btn.click(fn=convert_csv_to_parquet)

        demo.launch(height=1000)

    def add_to_train_set(self, df: pd.DataFrame, output_filename: str):
        """
        Add samples to the training set.
        """
        new_train_set = df.copy()

        logger.info(f"Adding {len(new_train_set)} samples to training set")
        self.train_set = pd.concat([self.train_set, new_train_set])

        self.train_set = self.train_set.drop_duplicates(
            subset=["filepath"], keep="last"
        )
        self.train_set.reset_index(drop=True, inplace=True)

        self.train_set.to_parquet(f"{output_filename}.parquet")
        logger.info(f"Saved training set to {output_filename}.parquet")
