# Fashion MNIST classification using CNN
## Problem
>

The main goal of this presentation is to construct simple convolutional neural networks which are capable of achieving approximately 95% accuracy without using some pretrained model for transfer learning.
We shall demonstrate this using different network architectures and combining several techniques in order to obtain relatively good results.

For this task we used TensorFlow 2.0 with integrated Keras API.

## Solutions
>
### Baseline Model

We started with simple CNN architecture which contained two convolutional layers with 3 by 3 kernels and 32 and 64 feature maps. We used ReLU activation function. Here, we applied the "same" padding since the Fashion MNIST images have only 28 x 28 pixels and "valid" padding would shrink images significantly. After each convolutional layer we applied batch normalization and max pooling layers. After the convolutional layers we used a dense layer with 512 neurons and a classification head with softmax for 10 labels. The model initially achieved training accuracy of about 90% on the training set.

In order to obtain better results we doubled the number of convolutional layers using two blocks of two convolutional layers. Henceforth in all cases we used 3x3 kernels, which gave us better results as well as 2x2 max pooling. We used two convolutional layers with 32 and 64 feature maps following the usual single dense layer with 512 neurons and classification head. The model achieved accuracy of about 91-92% on a test set and showed strong overfitting. In order to fix this problem we used dropout regularization. It turned out that small L2-regularization not only reduces overfitting, but also stabilizes the loss function convergence, making it smoother, according to our experience. After introducing dropout and L2-regularization the overfitting appears between the epoch 10 and 20 with approximately 93% accuracy. The accuracy on the test set reaches 94% at the epoch 30. The model performs better on the test set when there is a small amount of overfitting since the Fashion MNIST is very standardized, accordind to our opinion. The training time for this model is about 8 min using GPU, while the early stopping time would be at about half that time.

### Baseline model with data augmentation

We performed data augmentation using rotation around vertical axis and width and height variations of 10% since it yields the best results. Shear and rotation transformation decrease accuracy, possibly due to horizontal alignment of images in the dataset and diversification of that type makes it harder for model to classify the images.
Introducing data augmentation significantly reduced overfitting, while the accuracy on the test set reduced to 92-93%. The training time is significantly longer, about 50 min.

### Model with 3 convolutional blocks

We also tried a model with 3 convolutional blocks, adding two more convolutional layers with 128 feature maps,  with batch normalization, following one max pooling with dropout and L2-regularization. The model converges smoothly and achieves accuracy of about 92-93% at the 30th epoch. The training time is slightly longer than in the case of the baseline model. We trained it also on augmented data and it converges smoothly and best performance is achieved after 250 epochs and 1h 27min on GPU with accuracy of 92-93% on the test set.
