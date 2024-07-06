# Detecting Fake Speech Using Machine Learning
This project
## Introduction

This project, created by D. Arbiv and E. Mascareñas García, focuses on training a model capable of detecting fake audio, specifically fake speech, which may not be easily discernible by the human ear.

This endeavor is crucial for preventing misinformation, identity theft, audio-based fraud, and scams. Additionally, it enhances security measures, maintains the integrity of audio content, and protects against the misuse of deepfake technology. 

For any communications, you can reach us on our LinkedIn profiles: [D. Arbiv](https://www.linkedin.com/in/dror-arbiv/) and [E. Mascareñas García](https://www.linkedin.com/in/elena-mascare%C3%B1as-garc%C3%ADa-008615107/).


In this project, we aimed to train a model capable of detecting fake audio, specifically fake speech, which may not be easily discernible by the human ear. This endeavor is crucial for preventing misinformation, identity theft, audio-based fraud, and scams. Additionally, it enhances security measures, maintains the integrity of audio content, and protects against the misuse of deepfake technology.

## Project Overview
### Data Source
The dataset used in this project was obtained from [this link](https://deepfake-total.com/in_the_wild). Please refer to the original source for any licensing or usage restrictions. To download the dataset, follow these steps:

1. Visit [source link](https://deepfake-total.com/in_the_wild).
2. Download the dataset and extract the files to the data/ directory.
3. Change the names of each audio file based on the renaming_file.csv with converting_names.py file.
   
### Data Cleaning
We began with a dataset containing approximately 31,000 audio files. Our first task was to clean the data by removing noise and non-speech elements. This preprocessing step was essential to ensure the quality and relevance of the data for further analysis.

### Exploratory Data Analysis (EDA)
In the EDA phase, we utilized 13 Mel-Frequency Cepstral Coefficients (MFCCs) and Mel spectrograms to observe differences between AI-generated speech and real speech. We examined various attributes of the audio files, such as duration and speaker characteristics. The dataset included recordings of both historical figures from 100 years ago and contemporary celebrities. To uncover patterns and relationships within the data, we performed unsupervised learning techniques, such as clustering and Principal Component Analysis (PCA), to visualize the variance and identify distinguishing features.

### Modeling
We experimented with various machine learning models, including:

- XGBoost
- Logistic Regression with L1 regularization
- Random Forest
- K-Nearest Neighbors (KNN)
  
We also explored deep learning approaches, specifically **Convolutional Neural Networks (CNNs)**.

Among all models tested, XGBoost emerged as the best-performing model, achieving an overall accuracy of 90% and an F1 score of 92.3%.

### Feature Importance
To gain insights into the model's decision-making process, we analyzed feature importance. By examining the accuracy associated with each MFCC and the remaining features after applying L1 penalty, we identified that MFCC1 and MFCC2 were the most critical features for detecting fake speech.

### Conclusion
Our project successfully demonstrated the potential of machine learning models, particularly XGBoost, in detecting fake speech with high accuracy. The insights gained from feature importance analysis further highlighted the significance of specific MFCCs in distinguishing real from fake audio.

### Repository Structure
* data/: Contains the cleaned audio files and any relevant metadata.
* src/: Source code for data preprocessing, feature extraction, and model implementation.
* models/: Trained models and results of the experiments.
* README.md: Project overview and documentation.

### How to Run
1. Clone the repository:
```
git clone https://github.com/elenamascarenas-master/speech-project.git
cd detecting-fake-speech
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Download and extract the dataset to the data/ directory as described in the Data Source section.

4. Run the data preprocessing script:
```
python src/mfccs_files_creation.py
```

5. Train the models:
```
python src/basic_models.py
python src/CNN_model.py
```

### Future Work
* Experiment with other deep learning architectures, such as Recurrent Neural Networks (RNNs) and Transformers.
* Explore more sophisticated feature extraction techniques.
* Investigate real-time fake speech detection applications.
  
### Contributing
We welcome contributions to this project. Please fork the repository and submit a pull request with your changes.

### Acknowledgments
We would like to thank all contributors for this project and espicially for our supervisor Dr. Itai Dattner for his support and valuable feedback.






