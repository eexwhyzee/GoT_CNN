# Recognizing Game of Throne Images using Convolutional Neural Networks 

## Overview

I love Game of Thrones, but what I don't love are spoilers. During season 7 of
GoT, HBO suffered many episode leaks that resulted in a bunch of spoilers
floating around Instagram before episodes officially aired. 

Instagram has a discover page that is populated with content curated to the
user's interests, and if you frequently like GoT related content like me, the
discover page is bound to have unwanted GoT spoilers that the user may not want
to see. 

Inspired by this dilemma, I thought it'd be pretty cool to build a
Convolutional Neural Network that could recognize whether an image contains GoT
related content.

## Model

Initially, I was planning to use transfer learning through a model architecture
(such as Google's [Inception](https://arxiv.org/abs/1409.4842)) with
pre-trained weights. Instead, I decided to go a different route and build/train
a model from scratch since I was interested to seeing what kind of results it
could obtain. 

The CNN model is implemented using [Keras](https://keras.io/) and trained using
37,661 images and validated on 7,617 images on an AWS EC2 instance
(p2.xlarge). These images were downloaded directly from
Instagram's API using the [Instagram Scraper](https://github.com/rarcega/instagram-scraper). 

Afterwards, I tested the performance of the model using 1,774 new images that the
model has not seen yet. For more details about the model design/training workflow,
refer to `report/Capstone_Report.pdf` and `notebooks/GoT_CNN_2.0.ipynb`, both included in this repo.  

## Perfomance

The final CNN model had an accuracy of **81%** on the test dataset. Details can be
seen in the `GoT_CNN_eval.ipynb`.

![Confusion Matrix](https://i.imgur.com/QHL5rri.png)

For our purposes, I figured there's a higher tolerance for false positives
over false negatives. In terms of avoiding spoilers, a false negative will have
a bigger impact on the user in comparison to a false positve. 

Interestingly, I tested an image that I took myself of my girlfriend sitting on the Iron Throne that was on display at the AT&T store in San Francisco and the model wasable to recognize that the image did in fact contain GoT related content. Pretty Cool!

![](https://i.imgur.com/aAXo3rO.png)

## Getting Started

To test drive the model on your own computer, run this in your terminal to
install the requirements:

```
pip install -r requirements.txt
```
Then you will want to clone this repo and change directories:

```
git clone https://github.com/eexwhyzee/GoT_CNN.git
cd GoT_CNN/
```

## Usage

To run the model on the sample images included with this repo, simply run
`python GoT_app.py` in your terminal and the results will be outputted as seen
below (+ for GoT related content, - for Not GoT related content):

![Sample Output](https://i.imgur.com/1JHKIE7.png)

For the most part, the model does a pretty good job at classifying GoT related images, however, there are some weaknesses. For instance, as I suspected, the model was easily fooled by images with people with blonde hair (Kanye West and Taylor Swift), which is caused by the abundance of images of Daenerys Targaryen in the training data set, a popular character in GoT known for her blonde hair. To improve this, more Not GoT training data of people with blonde hair could be used to help increase the robustness of the model.

If you would like to test your own images, create a directory with the image(s)
and use the `-image_path` argument to point to the path of your
image(s). 

```
python GoT_app.py -image_path /PATH_TO_IMAGES/
```

There is also an option to use your own weights using the `-model_weights`
argument, the model will use the trained weights that I have included in
this repo by default.

## Resources

For more info on convolutional neural networks, I highly suggest Stanford's
CS231n course, which is available [online](http://cs231n.github.io/).




