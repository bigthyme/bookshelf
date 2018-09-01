+++
date = "2017-12-26T16:37:37-08:00"
description = "Classifying Numbers in Street View Images"
title = "SVHN Classification"
markup = "markdown"
+++

Given the Street View House Number dataset, train a model that can predict number classifications in similar images.

![svhn](https://i.imgur.com/ERm71gc.png)

### Implementation
There are numerous considerations to be made when building out a custom model. The thing to first consider is figuring out what data you're dealing with. In the SVHN databse, there are two types of datasets: dataset one being images containing number sequences and the dataset two, being a MNIST like dataset for single digit classification. Although the task was to identify a sequence of numbers, I ended up choosing the second dataset due to the simplicity of the data format. Given more time, I would've liked to retrain the first dataset. Expanding on my choice of dataset, I was mostly training on a CPU device and wanted to do as much as I could to reduce training time. Given that the choice dataset contained images that were `32 x 32 x 3`, I thought it'd be more advantageous to train on versus the alternative, `48 x 48 x 3`. Alongwith the actual images to be used in training, the corresponding labels needed to read into memory. Though a straight forward task, I made the decision to utilize one hot encoding for classifications. Here's the encoding scheme I used for the project:

```
# as opposed to the given labels
[1] // represents a 1
[10] // represents a 10

# added one last non-digit class and moved the 10 value to the 0th position
[1,0,0,0,0,0,0,0,0,0,0] // this now represents a 10
[0,0,0,0,0,0,0,0,0,0,1] // this now represents a non-digit classification
```

#### Preprocessing
To load the `.mat` data type, I utilized the `scipy` library to read the file into memory for processing. Initially, I loaded in both `train.mat` and `test.mat` until I found out that the `.fit` model in keras could split the training data by some percentage via the `validation_split` parameter. I also normalized every image to further alleviate the time spent on convolutions. Though I did my final training steps with `BGR` images, I did obtain comparable recents with greyscaled images. That being said, I'm not quite sure if it matters all the much which channels are used in training; so long as the predictions are processed in the same manner.

#### CNN Architecture
Every CNN starts with convolutional layers and end with a fully connected output layer. When building my own custom model, I used two conv layers accompanied by corresponding max pooling layers. Due to it's efficiency, I chose the `relu` activation function for all conv layers (e.g. `f(x) = max(0, x)`). `MaxPooling` was added to reduce spatial dimensionality for training as well as to control overfitting.
After adding those top layers, I added fully connected layers to accept the output from the max pooling layers. Due to training on a cpu device with limited resources, the largest fc layers I could acheive was `~1000` nodes. Then of course, since we're classifying digits we need 10 indices per each digit we want to predict. In my particular case, I added another layer (and corresponding dataset) to train an 11th class that would represent a digit/non-digit classification. The last layer is a `softmax` output that grabs the previous calculations and outputs values in a normalized form. Specifically, the numbers become ranges from 0 to 1 and add up to 1.

#### Training
Given the above model, I trained several times adjusting many different hyperparameters to improve accuracy and loss metrics. The first parameter I attempted to update was `batch_size` and `epochs`. I kept a few things constant after a few mishaps on early stages of training. Namely, having the right digit to non-digit ratio. In the end, I had `72081` non-digit images and `73254` digit images. I also split my training set where `30%` was allocated to the validation set. Here was my first run accuracy and loss:
![7](https://i.imgur.com/BNI1c8j.jpg)

I was able to acheive very high accuracies with just `7` epochs, though my best performing model given these constants was simply to bump the epoch from `7` to `25`. I also added an Early Stopping callback to the fit function to evaluate my best runs and stop if the loss function passed a certain threshold. I chose `25` due to running this model with `30` and having it stop at `26`, though the 26th run wasn't quite as accurate. I could've updated my early stopping parameters but the differences were seemingly minimal:
![25](https://i.imgur.com/nd0edh9.jpg)

Due to my CPU, I was unable to run the `VGG16` out of the box. The large `4096` output layers would consistently OOM my device and thus could not finish outputting graphs. Because of this, I ended up using a ported version of VGG  which did not yield good results with the same `hyperparameters` described above. *Simar results were acheived for the `VGG16` pretrained.*
![vgg](https://i.imgur.com/XtIjk4b.jpg)

#### References
```md
1. *Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks*
 Ian J. Goodfellow, Yaroslav Bulatov, Julian Ibarz, Sacha Arnoud, Vinay Shet - https://arxiv.org/abs/1312.6082
2. *On the Convergence of A Family of Robust Losses for Stochastic Gradient Descent*
 Bo Han, Ivor W. Tsang, and Ling Chen - https://arxiv.org/pdf/1605.01623.pdf
3. *VGG-16 pre-trained model for Keras*
 baraldilorenzo - https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
4. *Rectified Linear Units Improve Restricted Boltzmann Machines*
5. *Softmax*
Wikipedia - https://en.wikipedia.org/wiki/Softmax_function
6. *VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION*
Karen Simonyan and Andrew Zisserman - https://arxiv.org/pdf/1409.1556v6.pdf
```
