<<<<<<< HEAD
# Clinical-Notes-Classifier
=======
# Clinical Note Processing System

This project implements an automated system for processing clinical notes, featuring:
- Automatic classification of medical notes (e.g., Consultation, Discharge Summary)
- Text summarization of clinical notes
- REST API interface for easy integration

## Features
- Note Classification: Categorizes medical notes into different types
- Text Summarization: Generates concise summaries of clinical notes
- API Interface: Easy-to-use REST API for integration with other systems
- Configurable Models: Support for different transformer models

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API server:
```bash
python main.py
```

## API Usage
The system exposes the following endpoints:
- POST `/classify`: Classify a clinical note
- POST `/summarize`: Generate a summary of a clinical note

## Project Structure
>>>>>>> 40ac2ef (Initial commit: Upload clinical note classification project)
