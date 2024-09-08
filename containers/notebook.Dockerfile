# First stage: Build environment to install dependencies
FROM python:3.10-slim AS build

# Set the working directory in the build stage
WORKDIR /app

# Copy the requirements.txt file from the notebooks directory (one level up)
COPY ../notebook/requirements.txt /app/requirements.txt

# Install dependencies in the build stage
RUN pip install --no-cache-dir -r requirements.txt

# Second stage: Runtime environment for running Jupyter Notebook
FROM python:3.10-slim

# Set the working directory for the runtime stage
WORKDIR /app

# Copy only installed packages from the build stage
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Install Jupyter in the runtime stage
RUN pip install jupyter --no-cache-dir

# Copy the notebook directory (from one level up) into the runtime container
COPY ../notebook /app/notebook

# Expose Jupyter Notebook port
EXPOSE 8888

# Set default command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--notebook-dir=/app/notebook"]
