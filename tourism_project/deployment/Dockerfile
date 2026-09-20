
# Use a lightweight Python 3.11 image.
FROM python:3.11-slim

# Prevent Python from generating .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Display Python output immediately.
ENV PYTHONUNBUFFERED=1

# Define the home directory for the non-root user.
ENV HOME=/home/user

# Create the non-root user required for Hugging Face Spaces.
RUN useradd --create-home --uid 1000 user

# Set the application working directory.
WORKDIR $HOME/app

# Copy the dependency file first.
COPY --chown=user:user requirements.txt requirements.txt

# Upgrade pip and install the deployment dependencies.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit application.
COPY --chown=user:user app.py app.py

# Run the application as a non-root user.
USER user

# Hugging Face Docker Spaces use port 7860.
EXPOSE 7860

# Start the Streamlit application.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true"]
