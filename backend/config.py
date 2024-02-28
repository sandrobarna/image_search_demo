import os
from pathlib import Path

BASEPATH = Path(os.path.abspath(__file__)).parent

class AppSettings:

    QDRANT_URL: str = os.getenv('QDRANT_URL', 'localhost')
    QDRANT_PORT: int = int(os.getenv('QDRANT_PORT', 6333))

    MODEL_CACHE_DIR: str = str(BASEPATH / '.data/embedding_model')
    EMBEDDING_MODEL_BATCH_SIZE: int = 128
    QDRANT_UPLOAD_BATCH_SIZE: int = 128
    QDRANT_SEARCH_MAX_LIMIT: int = 100

    KNN_EXACT_SEARCH: bool = True

    IMAGE_DATA_PATH: str = str(BASEPATH / './.data/images')

if __name__ == '__main__':
    print(AppSettings.MODEL_CACHE_DIR)