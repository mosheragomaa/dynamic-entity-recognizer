import asyncio
import os
import time
from pathlib import Path
from typing import Literal

import aiofiles
import magic
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import create_model

load_dotenv()
api_key = os.environ.get("API_KEY")
client = genai.Client(api_key=api_key)

contents = []
prompt = (
    "I am going to show you pictures of some entities, for each picture, I will provide the entity's name."
    "Your first task is to learn which name belongs to which entity. "
    "After that, I will show you a new picture. Your second task is to tell me the names of any of my entities that you recognize in this new picture."
)
contents.append(prompt)


async def training_imgs_to_bytes(imgs_dir: list):
    """
    Convert a list of training image file paths into byte arrays.

    Args:
       imgs_dir (list[str | Path]): List of image file paths.

    Returns:
        list[bytes]: List of image bytes.
    """
    training_imgs_bytes = []
    for img in imgs_dir:
        async with aiofiles.open(img, "rb") as f:
            img_bytes = await f.read()
            training_imgs_bytes.append(img_bytes)
    return training_imgs_bytes


supported_mime_types = [
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
]


async def create_training_contents(entity_name: str, entity_directory_name: str | Path):
    """
    Prepare LLM training content for a given entity.

    Args:
        entity_name (str): Name of the entity.
        entity_directory_name (str | Path): Directory containing the entity's image files.
    Returns:
        list: List containing entity name and image parts.
    """
    training_contents = []
    training_contents.append(f"entity name:  {entity_name}")
    for byte in await training_imgs_to_bytes(entity_directory_name):
        mime_type = magic.from_buffer(byte, mime=True)
        if mime_type in supported_mime_types:
            training_contents.append(
                types.Part.from_bytes(data=byte, mime_type=mime_type)
            )
        else:
            print(f"Image extension not supported, image is going to be skipped..")
            continue
    return training_contents


def create_dynamic_model(entity_name="entity", entity_values=list[str]):
    """
    Create a dynamic Pydantic model with possible entity names.

    Args:
        entity_name (str): Field name for the entity.
        entity_values (list): Allowed entity values.

    Returns:
        BaseModel: Generated Pydantic model.
    """

    possible_entity_values = list(entity_values) + ["Unidentified"]

    return create_model(
        "DynamicModel", **{entity_name: (list[Literal[*possible_entity_values]], ...)}
    )


def retry(fn):
    """
    Decorator to retry a function up to 3 times with a 60s delay between attempts.
    Exceptions are re-raised after the third attempt.

    Args:
        fn (Callable): Function to wrap.

    Returns:
        Callable: Wrapped function with retry logic.
    """

    def wrapper(*args):
        for i in range(3):
            try:
                return fn(*args)
            except Exception as e:
                if i == 2:
                    raise e
                time.sleep(60)

    return wrapper


@retry
async def test(test_imgs_dir_name: str, training_contents, model):
    """
    Run entity recognition on test images using the trained model.

    Args:
        test_imgs_dir_name (str | Path): Directory containing test images.
        training_contents (list): Training data for the model.
        model (BaseModel): Pydantic model schema for responses.

    Returns:
        list: Parsed model predictions.
    """
    tasks = []

    if isinstance(test_imgs_dir_name, str):
        iterator = Path(test_imgs_dir_name).iterdir()
    else:
        iterator = test_imgs_dir_name

    for img in iterator:
        test_contents = []
        test_contents.append(training_contents)
        test_contents.append(
            "The following is the test image, is there any entity you know in it?"
        )
        async with aiofiles.open(img, "rb") as f:
            byte = await f.read()
        mime_type = magic.from_buffer(byte, mime=True)
        if mime_type in supported_mime_types:
            test_contents.append(types.Part.from_bytes(data=byte, mime_type=mime_type))
        else:
            print(f"Image extension not supported, {img} is going to be skipped..")
            continue
        task = client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=test_contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": model,
            },
        )
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    return [r.parsed for r in results]
