import hashlib
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, models, UpdateStatus, ScoredPoint

from backend.config import AppSettings
from backend.embeddings import TextImageEmbeddingModel, ClipTextImageEmbeddingModel

logger = logging.getLogger(__name__)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger().setLevel(logging.INFO)


def _is_image_file(filename: Path) -> bool:

    if not filename.is_file():
        return False

    return filename.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']


def _read_images(image_files: list[Path]) -> tuple[dict[str, np.ndarray], int]:
    """This function takes list of image file paths and reads them in dictionary (filename -> np.array)
    It also counts if some images are by chance corrupted and can't be read. Grayscale images are simply
    converted to RGB, by copying gray channel into 3 copies.

    :param image_files: List of image file paths to read.
    :return: Returns dictionary of read images (filename -> np.array) and integer - counting number of corrupted images.
    """

    images = {}
    n_failed = 0
    for fpath in image_files:

        try:
            img = Image.open(fpath)
        except Exception:
            n_failed += 1

        img = np.array(img)

        if img.ndim != 3:
            img = np.repeat(img[..., np.newaxis], 3, -1)  # grayscale interpolation into RGB

        images[fpath.name] = img

    return images, n_failed


class SearchEngine:

    def __init__(self, model: TextImageEmbeddingModel):
        """Defines abstraction layer for Text2Image semantic search service, includes methods for search and indexing.

        :param model: Text2Image embedding model
        """

        self.emb_model = model

        self.qdrant = QdrantClient(
            url=AppSettings.QDRANT_URL,
            port=AppSettings.QDRANT_PORT,
        )

    def _collection_exists(self, name: str) -> bool:
        try:
            self.qdrant.get_collection(name)
        except Exception:
            return False
        return True

    def create_collection(self, name: str) -> bool:
        """Creates collection with given name. Returns True if collection didn't exist and
        created successfully, False if it existed before.

        :param name:
        :return:
        """

        if self._collection_exists(name):
            return False

        self.qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=self.emb_model.embedding_dim,
                distance=Distance.DOT),  # we use normalized embeddings so this is technically cosine similarity
        )

        return True


    def add_images_from_folder(
        self,
        collection_name: str,
        path: Path | str
    ) -> bool:
        """This is helper method to populate given collection with image embeddings from the folder.
        Returns True if all found images from the folder have been indexed successfully, False otherwise.

        :param collection_name: The name of the collection to insert embeddings in. It should exist in advance.
        :param path: Path to the directory where image files reside.
        :return:
        """

        path = Path(path)

        if not path.exists():
            raise ValueError("Path doesn't exist.")

        if not path.is_dir():
            raise ValueError("Specified path must be a directory containing images.")

        if not self._collection_exists(collection_name):
            raise ValueError(f"Collection with name '{collection_name}' doesn\'t exist.")

        image_files = [p for p in path.iterdir() if _is_image_file(p)]

        logger.info(f'Found {len(image_files)} image files.')

        # process uploads in batches
        bs = AppSettings.QDRANT_UPLOAD_BATCH_SIZE
        total_batches = math.ceil(len(image_files) / bs)

        for n_batch, i in enumerate(range(0, len(image_files), bs)):

            images, n_failed = _read_images(image_files[i:i + bs])

            logger.info(f'Upserting batch: {n_batch + 1}/{total_batches}. '
                        f'Num corrupted images in batch: {n_failed}/{len(images)}')

            # filename hash is used as id, to avoid duplicates (e.g. when retrying)
            valid_images, valid_image_fns, valid_image_ids = list(zip(*[
                (img, fn, hashlib.md5(fn.encode('utf8')).hexdigest()) for fn, img in images.items()
            ]))

            valid_images_embs = self.emb_model.embed_image(valid_images, AppSettings.EMBEDDING_MODEL_BATCH_SIZE).tolist()

            res = self.qdrant.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=valid_image_ids,
                    vectors=valid_images_embs,
                    payloads=[{'filename': fn} for fn in valid_image_fns]
                ),
                wait=True
            )

            if res.status != UpdateStatus.COMPLETED:
                return False

        return True

    def search(
        self,
        collection_name: str,
        text_query: str,
        limit: int
    ) -> list[ScoredPoint]:
        """Performs KNN search of images using text query.

        :param collection_name: The name of the collection in which search will be done.
        :param text_query: Query string
        :param limit: How many top scored results to fetch.
        :return:
        """

        if not self._collection_exists(collection_name):
            raise ValueError(f"Collection with name '{collection_name}' doesn\'t exist.")

        if self.emb_model.count_tokens(text_query) > self.emb_model.max_text_tokens:
            raise Warning(f'Query token count exceeds max allowed limit of {self.emb_model.max_text_tokens}. '
                          f'Truncation will take place.')

        if limit > AppSettings.QDRANT_SEARCH_MAX_LIMIT:
            raise ValueError(f"limit value too big. Configured max is {AppSettings.QDRANT_SEARCH_MAX_LIMIT}")

        self.emb_model.embed_text(text_query)

        return self.qdrant.search(
            collection_name=collection_name,
            search_params=models.SearchParams(exact=AppSettings.KNN_EXACT_SEARCH),
            query_vector=self.emb_model.embed_text(text_query).tolist(),
            limit=limit,
            with_payload=True
        )

    def close(self):
        self.qdrant.close()

    def __del__(self):
        self.close()


def setup_demo():
    """This is helper function that spins-up demo by creating test collection and adding images from folder."""

    collection_name = 'demo_collection'

    logger.info(f'Initializing demo with settings: {json.dumps(AppSettings.as_dict(), indent=4)}')

    emb_model = ClipTextImageEmbeddingModel(AppSettings.MODEL_CACHE_DIR)

    search_client = SearchEngine(emb_model)

    if search_client.create_collection(collection_name):
        logger.info(f'Creating collection {collection_name} and starting index process...')
        if not search_client.add_images_from_folder(collection_name, AppSettings.IMAGE_DATA_PATH):
            raise RuntimeError("Adding images to collection failed.")

    return collection_name, search_client


