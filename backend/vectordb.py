import hashlib
import logging
import sys
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, UpdateStatus, models

from backend.embeddings import TextImageEmbeddingModel

logger = logging.getLogger(__name__)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger().setLevel(logging.INFO)


def is_image_file(filename: Path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    ext = filename.suffix.lower()
    return ext in image_extensions


class SearchEngine:

    def __init__(self, model: TextImageEmbeddingModel):

        self.emb_model = model
        self.qdrant = QdrantClient(url='localhost', port=6333)

    def _collection_exists(self, name: str) -> bool:
        try:
            self.qdrant.get_collection(name)
        except Exception:
            return False
        return True

    def create_collection(self, name: str):

        if self._collection_exists(name):
            raise ValueError(f"Collection with name '{name}' already exist.")

        return self.qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=self.emb_model.embedding_dim,
                distance=Distance.DOT),
        )

    def add_images_from_folder(
        self,
        collection_name: str,
        path: Path | str
    ) -> bool:

        path = Path(path)

        if not path.exists():
            raise ValueError("Path doesn't exist.")

        if not path.is_dir():
            raise ValueError("Specified path must be a directory containing images.")

        if not self._collection_exists(collection_name):
            raise ValueError(f"Collection with name '{collection_name}' doesn\'t exist.")

        # TODO images must fit in RAM
        image_files = [file_path for file_path in path.iterdir() if file_path.is_file() and is_image_file(file_path)]

        logger.info(f'Found {len(image_files)} images in total.')


        bs = 100
        total_batches = len(image_files) // bs + int(len(image_files) % bs > 0)

        for n_batch, i in enumerate(range(0, len(image_files), bs)):

            images = [np.array(Image.open(file)) for file in image_files[i:i+bs]]
            num_images = len(images)

            images = [i for i in images if i is not None and i.ndim == 3]
            num_corrupted_images = num_images - len(images)

            logger.info(f'Upserting batch: {n_batch + 1}/{total_batches}. '
                        f'Num corrupted images in batch: {num_corrupted_images}/{num_images}')

            # TODO take out hardcode
            image_embeddings = self.emb_model.embed_image(images, 32).tolist()

            _id = hashlib.md5().hexdigest()

            operation_info = self.qdrant.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=[str(uuid.uuid4()) for _ in range(len(images))],
                    vectors=image_embeddings
                ),
                wait=False
            )

            break

    def close(self):
        self.qdrant.close()

    def __del__(self):
        self.close()

