FROM python:3.12

# Set environment variables
ENV APP_HOME /app
ENV JUPYTER_PORT=1235
ENV FASTAPI_PORT=2222

# Set working directory
WORKDIR $APP_HOME

# Copy repo to docker location
COPY backend $APP_HOME/backend
COPY webapp $APP_HOME/webapp
COPY requirements.txt $APP_HOME/requirements.txt

# install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Command to run Jupyter and FastAPI servers
CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --port=$JUPYTER_PORT --allow-root --no-browser & cd webapp && uvicorn api:app --host 0.0.0.0 --port $FASTAPI_PORT"]


