import os
from pathlib import Path

BASEPATH = Path(os.path.abspath(__file__)).parent.parent

class AppSettings:
    """Defines app wide settings"""

    QDRANT_URL: str = os.getenv('QDRANT_URL', 'localhost')
    QDRANT_PORT: int = int(os.getenv('QDRANT_PORT', 6333))

    MODEL_CACHE_DIR: str = os.getenv('SEARCHAPP_DATA_BASEDIR', str(BASEPATH / 'data')) + '/embedding_model'

    # batch size that embedding neural network model will use in forward pass
    EMBEDDING_MODEL_BATCH_SIZE: int = 128

    # How many points to include in single upsert qdrant command
    QDRANT_UPLOAD_BATCH_SIZE: int = 128

    # max value of limit parameter of search query
    QDRANT_SEARCH_MAX_LIMIT: int = 100

    # whether to use exact or approximate search
    KNN_EXACT_SEARCH: bool = True

    # path to folder where images reside. This is used as a database for this project.
    IMAGE_DATA_PATH: str = os.getenv('SEARCHAPP_DATA_BASEDIR', str(BASEPATH / 'data')) + '/images'

    @staticmethod
    def as_dict() -> dict:

        x = dict(AppSettings.__dict__)

        keys_to_remove = [key for key in x.keys() if key.startswith('_')]
        for key in keys_to_remove:
            del x[key]

        del x['as_dict']

        return x

