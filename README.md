# Traffic Sign Recognition

---

**Build a Traffic Sign Recognition Project**

Project steps:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

---

### Data Set Summary & Exploration

#### Dataset Summary

- Number of training examples = 34799
- Number of testing examples = 12630
- Number of validation examples = 4410
- Image data shape = (32, 32, 3)
- Number of classes = 43


#### Exploratory Visualization

Distribution of Training Examples
![Histogram](images/hist.png)



### Design and Test a Model Architecture

#### Preprocessing

- Normalize image data
- Shuffle training dataset

#### Model Architecture

The final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Relu					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6  	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| Relu					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16  	|
| Fully connected		| Outputs 120        							|
| Relu  				| Dropout 0.5        							|
| Fully connected   	| Outputs 84									|
| Relu                  | Dropout 0.5									|
| Fully connected		| Outputs n_classes		    	         		|
| Softmax               | 												|


#### Model Training

- Learning rate: 0.001
- Epochs: 20
- Bach size: 256
- Dropout: 0.5

#### Solution Approach

The final model results were:
* training set accuracy of 0.988
* validation set accuracy of 0.952
* test set accuracy of 0.929

##### First Iteration
LeNet architecture based on the previous lab.

![alt text](images/lenet.png)

* training set accuracy > 0.99
* validation set accuracy < 0.90

Result:  **Overfitting**


##### Second Iteration
Add Dropout of 50% on the fully connected layers.

![alt text](images/lenet-drop.png)

* training set accuracy of 0.988
* validation set accuracy of 0.952

Result: **Validation accuracy > 0.93**

### Test a Model on New Images

#### Acquiring New Images

German traffic signs on the web:

![alt text](test_images/12.jpg) ![alt text](test_images/14.jpg) ![alt text](test_images/15.jpg) ![alt text](test_images/17.jpg)
![alt text](test_images/18.jpg) ![alt text](test_images/3.jpg)
![alt text](test_images/32.jpg) ![alt text](test_images/9.jpg)



#### Performance on New Images

Results of the prediction:


| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Priority road      		| Priority road  									|
| Stop     			| Stop 										|
| No vehicles					| No vehicles											|
| No entry	      		| Stop					 				|
| General caution		| General caution      							|
| Speed limit (60km/h)		| Speed limit (60km/h)      							|
| End of all speed and passing limits		| End of all speed and passing limits     							|
| No passing		| No entry      							|

Accuracy of 75%

#### Model Certainty - Softmax Probabilities

![Softmax](images/softmax.png)
