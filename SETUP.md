# Drug Discovery Toolkit - Setup Guide

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (for ArangoDB)
- At least 8GB of RAM
- (Optional) NVIDIA GPU with CUDA support for faster inference

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Drug-Discovery-Toolkit.git
cd Drug-Discovery-Toolkit
```

### 2. Set Up Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Set Up ArangoDB Database

Start ArangoDB using Docker Compose:

```bash
docker-compose up -d
```

This will:
- Start ArangoDB on port 8529
- Set the root password to "openSesame"
- Enable vector index support
- Create persistent storage for the database

The database and collections will be automatically created when you first run the application.

### 4. Configure Environment Variables

Copy the example environment file and update it with your credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your Google API key:
- Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Replace `your_google_api_key_here` with your actual key

### 5. (Optional) Download TamGen Model

For molecular generation features, you need to download the TamGen checkpoint:

1. Download the model from [this link](https://microsoftapc-my.sharepoint.com/:f:/g/personal/v-kehanwu_microsoft_com/EipAXgQfu6lPm1y2OP1ZUyEBsqQbPZ7aukhJ8_hgUej0yw?e=fE9G6h)
2. Create the checkpoints directory:
   ```bash
   mkdir -p checkpoints/crossdock_pdb_A10
   ```
3. Place `checkpoint_best.pt` in `checkpoints/crossdock_pdb_A10/`

Note: The application works without this model, but molecular generation features will be disabled.

### 6. Run the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Troubleshooting

### Database Connection Issues

If you see "database not found" errors:
1. Ensure Docker is running
2. Check if ArangoDB container is running: `docker ps`
3. Wait a few seconds for the database to initialize
4. The database will be automatically created on first run

### Import Errors

If you see import errors:
1. Make sure you've activated the virtual environment
2. Reinstall dependencies: `pip install -r requirements.txt`
3. For TamGen-related errors, ensure you're in the project root directory

### GPU/CUDA Issues

If you want to use CPU only:
- The application automatically falls back to CPU if CUDA is not available
- You can force CPU usage by setting in your `.env`:
  ```
  CUDA_VISIBLE_DEVICES=""
  ```

### Memory Issues

If you encounter memory errors:
- Reduce batch sizes in the model configurations
- Consider using a machine with more RAM
- For TamGen, you can reduce the beam size and max tokens

## Features Available Without Optional Components

Even without the TamGen model, you can still:
- Search for drugs and proteins in the database
- Analyze drug-protein interactions
- View molecular structures in 2D and 3D
- Calculate molecular properties
- Get ChemBERTa embeddings
- Use the AI chat interface for drug discovery queries

## Development Setup

For development with better performance:

```bash
pip install watchdog  # For auto-reload
```

Run with debug mode:
```bash
streamlit run app.py --server.runOnSave true
```

## Data Population

To populate the database with drug and protein data:
1. Place your data files in the `data/` directory
2. Use the provided import scripts (if available)
3. Or use the application's data import features

## Security Notes

- Never commit your `.env` file to version control
- Keep your Google API key secure
- Change the default ArangoDB password in production
- Use HTTPS in production deployments

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the error messages in the console
- Check the ArangoDB web interface at `http://localhost:8529`
- Create an issue on the project's GitHub repository