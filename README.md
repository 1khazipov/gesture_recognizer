# Team members
Anvar Iskhakov \
email: _an.iskhakov@innopolis.univerisity_

Albert Khazipov \
emial: _a.khazipov@innopolis.university_ 

Dmitrii Naumov \
email: _d.naumov@innopolis.university_

# Prerequisites

### Dependencies
Install all required packages with
```bash
pip install -r requirements.txt
```

# Datasets
### Original dataset
Download dataset from
https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed 

And unzip in folder `data/raw/dataset`

### Custom dataset
If you want to create your own dataset you can record videos for your own classes and put them in `data/raw/custom_video_dataset`. Video of each class should be located in the folder of it's class name. E.g. videos for class `plane` should be stored in `data/raw/custom_video_dataset/plane`

# Prepare Data
To prepare wlasl dataset for further pre-processing enter following command for the repository root:
```bash
python src/data/dataset_preprocessing.py 
```
To prepare custom dataset for further pre-processing enter following command for the repository root:
```bash
python src/data/custom_dataset_preprocessing.py 
```
To pre-process merged dataset to further training enter following command for the repository root:
```bash
python src/data/video_keypoints_extractor.py 
```

# Training
Currently available only in `asl_recognition.ipynb` notebook