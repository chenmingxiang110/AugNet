# AugNet

Please download the STL10 dataset from:
https://cs.stanford.edu/~acoates/stl10/
and put the files under "./data/stl10_binary".

Please download the pretrained model from:
https://drive.google.com/file/d/1pV3EBZPDDc3z_YKdRJu6ZBF5yn_IHhsK/view?usp=sharing
and put the pth file under "./models"

Run "res34_model_training_with_STL.py" to train a model. The pretrained model is trained with STL10 unlabeled data only. Due to file size limitations, we can only provide one pre-trained model. But in general, all test results are obtained using the same training method, so the training data and the model's backbone are the only things needed to be changed (different types of backbones can be viewed in "./lib/utils_torch.py").

Run "test_with_STL_kmeans.py" to test with K-Means clustering.
