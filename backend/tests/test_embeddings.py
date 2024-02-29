import pytest
import torch
from PIL import Image

from backend.config import AppSettings
from backend.embeddings import ClipTextImageEmbeddingModel

@pytest.fixture
def model():

    model = ClipTextImageEmbeddingModel(cache_dir=AppSettings.MODEL_CACHE_DIR)

    return model


def test_emb_model_shapes_match(model):

    image = Image.open('mock_data/seven_up_poster.jpg')

    img_vector = model.embed_image([image, image], batch_size=1)

    text_vector = model.embed_text(
        ['this is sample text 1',
         'this is sample text 2',
         'this is sample text 3'],
        batch_size=2
    )

    assert len(text_vector) == 3 and all(len(v) == model.embedding_dim for v in text_vector)

    assert len(img_vector) == 2 and all(len(v) == model.embedding_dim for v in img_vector)


def test_emb_model_semantic_similarity(model):


    image = Image.open('mock_data/seven_up_poster.jpg')

    image_embedding = model.embed_image([image])
    query_embeddings = model.embed_text([
        'image with green background',
        'image of 7 up',
        'image of soda drink',
        'image with red background'
    ])

    cos_similarities = torch.matmul(query_embeddings, image_embedding.t())

    _, indices = torch.topk(cos_similarities, k=cos_similarities.shape[0], dim=0)

    assert indices.tolist() == [[1], [0], [2], [3]]