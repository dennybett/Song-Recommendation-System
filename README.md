# BCS_Project_2 - Spotify Song Recommendation System

## Overview
This project is a **Song Recommendation System** that utilizes **Unsupervised Machine Learning clustering algorithms** to recommend songs based on user-selected audio features. The system is built in conjunction with **Streamlit** for the users' interface and employs various clustering techniques such as **K-Means, Agglomerative Clustering, and Gaussian Mixture Models** to group songs with similar characteristics.

## Features
- **User Input via Streamlit UI**: Users can select various song characteristics like danceability, energy, and tempo to find the ideal songs.

- **Machine Learning**: We employed Unsupervised ML models to cluster the songs based on their features.

  - ***K-Means Clustering***: Groups songs based on feature similarities.
  - ***Agglomerative Clustering***: Hierarchical clustering for music classification.
  - ***Gaussian Mixture Model(GMM)***: Probabilistic model for clustering.

- **Pre-trained Models**: The project loads pre-trained models to make song recommendations efficiently.

- **Data Preprocessing**: Transforms the raw data into a clean and usable format, to ensure that machine learning algorithms effectively analyzes and learn from the data, ultimately improving model performance and accuracy.
  - Scaling (StandardScaler, MinMaxScaler)
  - Variance Inflation Factor (VIF) Analysis
  - Exploratory Data Analysis (EDA) functions

### Model and Scaling Choice
After testing multiple scaling methods (StandardScaler, MinMaxScaler) and clustering models (K-Means, Agglomerative Clustering, Gaussian Mixture Model), we ultimately selected **K-Means clustering** with **MinMax scaling** for the final output. The Elbow Curve and the Calinski-Harabasz score were used as evaluation metrics, which reinforced that this combination provided the best performance and accuracy for the music recommendation system.
.

## Installation
### **1. Clone the Repository**
```bash
git clone https://github.com/Jeff-Oliver/BCS_Project_2.git

cd BCS_Project_2
```

### **2. Install Dependencies**
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels streamlit joblib
```

### **3. Run the Streamlit App**
```bash
streamlit run interface.py
```

### **4. Verify Installation**
Run the following command to check if Streamlit is installed correctly:
```bash
streamlit --version
```
If everything is set up correctly, you should see the installed version of Streamlit.

## Program Usage Guide
1. **Run `main.ipynb` first** to ensure all necessary data is processed correctly before using the recommendation system.
2. **Launch the Streamlit Web App**.
3. **Use the sidebar sliders** to set song characteristics (e.g., danceability, energy, tempo) depending on your preference.
4. Click **"Find My Song"** to get songs recommendation.
5. The system will return **up to 10 similar songs** based on the selected algorithm.

## Example output

Below is an example of the expected Streamlit UI when a user searches for songs:

![alt text](<Screenshot 2025-02-18 at 8.46.11 PM-1.png>)

## File Structure
```
                BCS_Project_2
│── data                  # CSV dataset of the songs used in the ML model 

│── output                # Contains pre-trained ML models & processed datasets

│── interface.py          # Streamlit UI & Recommendation Logic

│── main.ipynb            # Jupyter Notebook for data exploration & modeling (MUST be run first)

│── README.md             # Project Documentation (This File)

│── utils.py              # Utility functions for EDA & preprocessing

│── Project2 v3 Demo.mp4  # Video demonstration of the project

│── presentation.pptx     # PowerPoint presentation for project showcase
```

## Project Demonstration

A video demonstration (`Project2 v3 Demo.mp4`) is included in the project directory to provide a walkthrough of the system. This video showcases:

- An overview of the project

- How to interact with the Streamlit UI

- The machine learning models in action

- Example song recommendations


## Powerpoint Presentation

A **PowerPoint presentation (`presentation.pptx`)** is included in the project directory to help explain the system to interested parties. This presentation covers:
- Project objectives and motivation
- Overview of the machine learning models used
- Key insights from data analysis
- Demonstration of the recommendation system

## Troubleshooting

**Streamlit App Won't Start**

- Solution: Ensure the Streamlit dependency has been correctly installed and verify its installation. Try running:

```bash
pip install streamlit
```
```bash
streamlit --version
```

**Missing File Errors**

- Solution: Make sure you have run `main.ipynb` before launching `interface.py`. This generates all the necessary datasets required to run the program.

## Project contributors

- Saman Zahra Raza
- Mark Murphy
- Leonardo Rastelli Galebe
- Steven Frasica
- Jeff Oliver 
- Dennis Kipng'eno Bett

## Acknowledgments

This project utilizes the following datasets and documentations:
1. **Top 10000 songs EDA & Models**: Available on [Kaggle](https://www.kaggle.com/code/joebeachcapital/top-10000-songs-eda-models/notebook), this dataset provides a comprehensive collection of songs with sentiment analysis.

2. [Python Documentation](https://docs.python.org/3/).

3. [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/index.html).

4. [Scikit-learn Documentation](https://scikit-learn.org/stable/index.html).

5. [Referenced article for creating an AI-based music recommendation system](https://www.eliftech.com/insights/all-you-need-to-know-about-a-music-recommendation-system-with-a-step-by-step-guide-to-creating-it/).

## Contribution Guideline

Contributions are welcome and appreciated! Please follow these steps:
1. **Fork** the repository.
2. Create a **feature branch** (`git checkout -b feature-branch`).
3. **Commit your changes** (`git commit -m "Added new feature"`).
4. **Push to GitHub** (`git push origin feature-branch`).
5. Open a **Pull Request**.

## License
This project is licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Contact
For questions, suggestions, or contributions, kindly reach out via GitHub Issues.

