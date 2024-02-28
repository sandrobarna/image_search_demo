import base64
import json
import sys

from starlette.responses import FileResponse

sys.path.append('../')

from fastapi import FastAPI
from backend.vectordb import setup_demo
from backend.config import AppSettings

from pydantic import BaseModel, Field, constr, conint
from fastapi.staticfiles import StaticFiles

class UserQuery(BaseModel):

    # here maxlength 100 is just picked randomly to protect server from very long inputs
    # that could overload embedding model backend
    # embedding model might do its own truncation, see backend logic for details.
    text: constr(
        strip_whitespace=True,
        min_length=1,
        max_length=100,
        to_lower=True
    ) = Field(..., description="Text query")

    limit: conint(
        ge=1, le=AppSettings.QDRANT_SEARCH_MAX_LIMIT
    ) = Field(1, description="How many top results to fetch.")


app = FastAPI()


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:1234",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
# here we prepare web server for the demo use case.
collection_name, search_client = setup_demo()


def encode_image_base64(image_path: str):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@app.get("/")
def root():
    return FileResponse('static/index.html')

@app.post("/query/")
def query(query: UserQuery):

    res = search_client.search(
        collection_name,
        query.text,
        query.limit
    )

    res = [{
        'image_base64': encode_image_base64(
            AppSettings.IMAGE_DATA_PATH + '/' + i.payload['filename']
        ),
        'score': i.score
    } for i in res]

    return json.dumps(res)