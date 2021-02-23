# Automatic-Garbage-detection-system
<hr>

Following are details of our  progress regarding the SSIP Project (Garbage Detection).

The dataset that we used to test our models was the TACO datset.

1) We focused on improving the previous implementation of the Machine Learning model.  We tried some normal deep learning models with the layers ranging from 5-15 containing around 200-300 neurons each with some dropout layers and also tried some Batch Normalization (to reduce overfitting).

2) After trying all of these architectures and modifications, we observed no real progress. The accuracy on almost all of these architectures varied between 85-87%. Therefore, we decided to try out some SOTA architectures like Resnet-50.

3) After training the pre-trained Resnet-50 model, we gained a huge 3% accuracy boost on the training set, reaching 90% accuracy, which is still low.

4) Thereafter, we used the weights of the ImageNet database to initialize our model's weights. We re-trained the Resnet-50 model. Fortunately, we gained 2% accuracy boost, finally achieving 92% accuracy. 

5) Next, we are going to perform Image Segmentation on the TACO dataset. Moreover, we will capture some pictures of the garbage, so that the dataset's size increases, so we can try out other existing SOTA models.

6) We are also thinking of using opencv library to extract the images of garbage from the video clips so that we can feed these images to the model to classify them. Once the model is finalised, we are going to integrate it with the hardware.

![](https://github.com/BhavsarCoding/Automatic-Garbage-detection-system/blob/main/2020-12-15_16_53_12-Window.png)

![](https://github.com/BhavsarCoding/Automatic-Garbage-detection-system/blob/main/2020-12-15_16_53_50-Window.png)

![](https://github.com/BhavsarCoding/Automatic-Garbage-detection-system/blob/main/2020-12-15_16_54_27-Window.png)
