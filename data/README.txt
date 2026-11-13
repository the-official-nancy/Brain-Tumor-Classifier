Put the dataset in this folder so it looks like this:

data/
  Training/
    glioma/       *.jpg
    meningioma/   *.jpg
    notumor/      *.jpg
    pituitary/    *.jpg
  Testing/
    glioma/ meningioma/ notumor/ pituitary/

Get it automatically:   python download_dataset.py
Or download manually:   https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Check it:               python dataset.py --data_dir data
