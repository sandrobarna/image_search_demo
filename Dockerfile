# Use Python base image
FROM python:3.12

# Set environment variables
ENV APP_HOME /app
ENV JUPYTER_PORT=8888
ENV FASTAPI_PORT=1234

# Set working directory
WORKDIR $APP_HOME

# Copy repo to docker location
COPY backend $APP_HOME/backend
COPY webapp $APP_HOME/webapp

# Install requirements
#RUN pip install --no-cache-dir -r requirements.txt

RUN pip install jupyter && \
    pip install --quiet --no-cache-dir \
    'torch==2.2.1' \
    'transformers==4.38.1' \
    'qdrant-client==1.7.3' \
    'pillow==10.2.0' \
    'uvicorn==0.27.1' \
    'fastapi==0.110.0'

# Command to run Jupyter and FastAPI servers
CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --port=$JUPYTER_PORT --no-browser --allow-root"]


