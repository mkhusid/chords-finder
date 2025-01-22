# Music Chords Analysis Platform

# Music Chords Analysis Platform

This repository contains a set of Jupyter notebooks and Python modules for various music analysis tasks. 
<br/>Below are the details and instructions on how to use each component.

## Author

Maintained by **Maksym Khusid**, a Software Engineer and MA student of Kyiv Polytechnic Institute.

## General Info

This project is related to academia coursework and diploma research in Big Data and ML fields.
<br/>The idea is to build platform for musicians, music analysts, and researchers looking to explore the harmonic, rhythmic, and structural elements of songs.
<br/>By leveraging Python libraries like Librosa, PySpark capabilities for scaling and real-time streaming and machine learning models.

## Prerequisites

Make sure you have the following installed:

- Python 3.x
- Jupyter Notebook
- Required Python packages (listed in `requirements.txt`)
- Apache Spark

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Important Notes

- You DON'T need to run python modules seperatly from notebooks.
- Use notebooks in this order: chords_analyzer -> genre_analyzer -> songs_finder

## Notebooks

### 1. **chords\_analyzer.ipynb**

This notebook allows to analyze music fragment that is being recorded in real-time to detect chord progressions.
It provides visualizations and statistical analysis of chord progressions.

**Features:**
- Detect chords from real-time audio streams using the Librosa library.
- Generate timestamps for chord transitions.
- Use a pre-filtering mechanism to smooth the record and filter out missdetected chords.
  
**Usage:**
- Start recording your musical fragment.
- Chord detection algorithm will be initialzed on recording.
- Analyze and visualize the detected chords.

### 2. **genre\_analyzer.ipynb**

This notebook classifies songs into different genres based on their audio features.<br/> It uses machine learning models to predict the genre of a song.

**Features:**

- Preprocess Spotify audio features dataset.
- Train and evaluate machine learning models (e.g., Random Forest, Logistic Regression).
- Classify new songs based on their audio features.

**Usage:**

- Load the Spotify dataset with labeled genres.
- Train the model using the training pipeline.
- Predict genres for new songs by providing their audio features.

### 3. **songs\_finder.ipynb**

This notebook helps in finding songs based on specific criteria such as tempo, key, or mood. It uses audio feature extraction to filter songs.

**Features:**

- Filter songs using criteria like tempo, energy, and danceability.
- Perform advanced searches using multiple feature thresholds.

**Usage:**

- Load dataset of Spotify songs.
- Define filtering criteria in the notebook.
- Get a list of songs matching the specified criteria.

### 4. **chroma\_exploration.ipynb** - ONLY FOR RESEARCH

This notebook is only for exploration features of "librosa" package for build chroma vectors. <br/>
Chroma features represent the 12 different pitch classes and are useful for analyzing harmonic content.

**Features:**

- Extract chroma features from audio data.
- Compare chroma features across different computation algorithms using visualisation.

**Usage:**

- Load your audio files & use the provided functions to compute and visualize chroma features.

## Python Modules

### 1. **chords\_detection.py**

A module for detecting and analyzing chords in audio files.<br/>
The core of chords-detection logic.

**Features:**

- Chord detection using chroma features.
- Generating timestamps for chord changes.
- Smoothing and refining chord transitions.

### 2. **feature\_recognizer.py**

A module for extracting and analyzing audio features.

**Features:**

- Chroma feature extraction.
- Tempo, evergy, valence and other abstract features detection.
- Preparing feature vectors for machine learning models.


## Project Structure

- `notebooks/`
  - Contains the Jupyter notebooks.
- `src/`
  - Contains the Python modules for reusable functionality.
- `requirements.txt`
  - Lists the required Python packages.
- `data/`
  - Placeholder for input datasets or audio files.
- `outputs/`
  - Stores outputs such as visualizations or processed data.
