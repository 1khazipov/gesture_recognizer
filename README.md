# Team members
Anvar Iskhakov \
email: _an.iskhakov@innopolis.univerisity_

Albert Khazipov \
email: _a.khazipov@innopolis.university_ 

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

# Train model
To train any model on the preprocessed dataset enter following command for the repository root (example for a simple lstm model):
```bash
python src/models/simple_lstm/train_model.py
```

# Inference
To use the final trained model on your own videos enter following command for the repository root (example for a simple lstm model):
```bash
python src/models/simple_lstm/predict_model.py --file_path 'path/to/video.mp4'
```

# Miscellaneous
One can try models in action (live) by running the following command:
```bash
python demo.py
```
One can add some arguments as well. Command with default arguments is (example for a simple lstm model):
```bash
python demo.py --threshold 0.9 --checkpoint_name 'best.pt' --model_name 'simple_lstm' 
```
