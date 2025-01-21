# Music Chords Analysis Platform

This repository contains a set of Jupyter notebooks for various music analysis tasks. Below are the details and instructions on how to use each notebook.

## Notebooks

1. **chords_analyzer.ipynb**
2. **chroma_exploration.ipynb**
3. **genre_analyzer.ipynb**
4. **songs_finder.ipynb**

### Prerequisites

Make sure you have the following installed:
- Python 3.x
- Jupyter Notebook
- Required Python packages (listed in `requirements.txt`)

You can install the required packages using:
```bash
pip install -r requirements.txt
```

### Notebooks Description:

1. **chords_analyzer.ipynb**  
    This notebook analyzes the chords used in a given set of songs. It provides visualizations and statistical analysis of chord progressions.

    **Usage:**
    - Record 

2. **chroma_exploration.ipynb**  
    This notebook explores the chroma features of songs. Chroma features represent the 12 different pitch classes and are useful for analyzing harmonic content.

    **Usage:**
    - Load your audio files.
    - Extract chroma features using the provided functions.
    - Visualize the chroma features.

3. **genre_analyzer.ipynb**  
    This notebook classifies songs into different genres based on their audio features. It uses machine learning models to predict the genre of a song.

    **Usage:**
    - Load Spotify dataset with labeled genres.
    - Train the model using the provided cells.
    - Predict the genre of new songs.

4. **songs_finder.ipynb**  
    This notebook helps in finding songs based on specific criteria such as tempo, key, or mood. It uses audio feature extraction to filter songs.

    **Usage:**
    - Load your dataset of songs.
    - Define the criteria for finding songs.
    - Run the cells to get the list of songs matching the criteria.
