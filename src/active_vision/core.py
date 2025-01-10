from loguru import logger
from fastai.vision.models import resnet18, resnet34


def load_model(model_name: str):
    logger.info(f"Loading model {model_name}")
    if model_name == "resnet18":
        return resnet18()
    elif model_name == "resnet34":
        return resnet34()
    else:
        logger.error(f"Model {model_name} not found")
        raise ValueError(f"Model {model_name} not found")
