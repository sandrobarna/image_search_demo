# Image Search using Text

## Description
This is demo project showcasing image similarity search using text queries. Project uses [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) 
model for text/image embeddings [Qdrant](https://qdrant.tech) vector database for KNN search.

## Architecture

The app contains following components:
 - Qdrant service (runs locally)
 - Similarity search backend (see backend folder in repo) that computes embeddings and communicates with qdrant service
 - FastAPI based REST API and HTML frontend 
   - For demo purposes, functionality is limited to K-NN search only (adding/editing existing images isn't supported from web)

## How to Use

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/sandrobarna/image_search_demo.git
   
2. **Download Data**:

Embedding model weights, test images as well as Qdrant's collection pre-populated with those image vectors need to be
downloaded and mapped to respective docker volumes. Download it from [HERE](https://drive.google.com/file/d/14Taqf7ds1Ccegi5ciPDhQFnDU-ozYbYc/view?usp=share_link).



3. **Spin-up services**:

   There is a docker-compose file in repo which spins up Qdrant service (default port 6333), jupyter notebook (default
port 1235) for playing with exploratory data analysis (backend/notebooks/exploratory_analysis.ipynb) and FastAPI server
(default port 2222) for using HTML frontend.<br /><br />

   You need to map `embedding_model` and `images` subfolders (from the data downloaded in step 2) to `/searchapp_data` docker
volume as shown below (in docker-compose.yaml).

   ```docker
    volumes:
      - ./data/embedding_model:/searchapp_data/embedding_model # embedding model weights
      - ./data/images:/searchapp_data/images # folder containing images
   ```
   Now map Qdrant's storage folder `qdrant_storage` (also from the data downloaded in step 2)

   ```docker
   volumes:
      - ./data/qdrant_storage:/qdrant/storage # qdrant storage (already contains demo collection pre-populated with images)
   ```

   <br />   
   Modify default ports for jupyter or webapp if you wish in docker-compose.yaml
   <br /><br />
   Once all set, run following to fire up:

   ```bash
   docker-compose up
   ```
   
3. **Access Web App**:

   Open web browser and go to one of the following, as you wish:
    - `localhost:2222` to try out demo webapp
    - `localhost:1235` to go to Jupyter. You can play with backend/notebooks/exploratory_analysis.ipynb in order to see some
good/bad query examples as well as some data stats. **Jupyter secret token can be copied from docker-compose logs printing in terminal**
    - `localhost:6333/dashboard` to go to Qdrant's dashboard


