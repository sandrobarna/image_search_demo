from abc import ABC, abstractmethod
from typing import Callable
from typing import Union, List, TypeVar, Tuple

import numpy as np
import torch
from transformers import CLIPModel, AutoProcessor, AutoTokenizer

T = TypeVar('T')
OneOrMany = Union[T, List[T], Tuple[T]]


class TextImageEmbeddingModel(ABC):
    """Abstract class for Text2Image Embedding models"""

    @abstractmethod
    def embed_text(self, texts: OneOrMany[str], batch_size: int | None = None) -> OneOrMany[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def embed_image(self, images: OneOrMany[np.ndarray], batch_size: int | None = None) -> OneOrMany[torch.Tensor]:
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def max_text_tokens(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        raise NotImplementedError


class ClipTextImageEmbeddingModel(TextImageEmbeddingModel):

    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        cache_dir: str
    ):
        """CLIP Vision Transformer based Image2Text embedding model.


        :param cache_dir: Folder where model weights are downloaded.
        """

        self._model = CLIPModel.from_pretrained(self.MODEL_ID, cache_dir=cache_dir)
        self._image_processor = AutoProcessor.from_pretrained(self.MODEL_ID, cache_dir=cache_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, cache_dir=cache_dir)


    @staticmethod
    def _calc_embeddings(
            data: list[str | np.ndarray],
            preprocessor: Callable,
            emb_func: Callable,
            batch_size: int | None
    ) -> torch.Tensor:

        embeddings = []

        if batch_size is None:
            batch_size = len(data)

        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i+batch_size]
                batch_data_features = preprocessor(batch_data)
                embeddings.append(emb_func(**batch_data_features))

        embeddings = torch.cat(embeddings, dim=0)

        return embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

    def embed_text(self, texts: OneOrMany[str], batch_size: int | None = None) -> OneOrMany[torch.Tensor]:

        is_many = isinstance(texts, (list, tuple))

        if not is_many:
            texts = [texts]

        res = self._calc_embeddings(
            texts,
            lambda x: self._tokenizer(text=x, padding=True, return_tensors="pt", truncation=True),
            self._model.get_text_features,
            batch_size
        )

        if not is_many:
            res = res[0]

        return res


    def embed_image(self, images: OneOrMany[np.ndarray], batch_size: int | None = None) -> OneOrMany[torch.Tensor]:

        is_many = isinstance(images, (list, tuple))

        if not is_many:
            images = [images]

        res = self._calc_embeddings(
            images,
            lambda x: self._image_processor(images=x, return_tensors="pt"),
            self._model.get_image_features,
            batch_size
        )

        if not is_many:
            res = res[0]

        return res

    @property
    def embedding_dim(self) -> int:
        return 512

    @property
    def max_text_tokens(self) -> int:
        return self._tokenizer.model_max_length

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.tokenize(text=text, add_special_tokens=True))

