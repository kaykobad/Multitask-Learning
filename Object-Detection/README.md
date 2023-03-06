## Baselines model for Unicorn dataset

In this zip file, you'll find a simple 
PyTorch script that can train on the 
Unicorn dataset with a ResNet50 model. If 
you want to change the model, you can. In 
the code, you'll find some arguments. 

### Training
To train, simply pass in the train_folder argument (the path to the training images). From the command line, it will look like ```python3 simple_baselines.py --train_folder "path/to/training/images"```
You can also specify the batch size, the number of epochs to train for, 
the number of workers etc. but there are default values already there.

### Testing 
To test, specify the flag ```test_only``` and pass in the training 
images and the path to your model which should have been saved from training. 


That's it! Remember, this is a baselines model and doesn't 
have any data augmentation or anything special to improve training. 
That's on you now! Good luck!