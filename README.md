# SkinRecognition
## Introduction
Three sets of images have been provided for the Skin Recognition project:
# Training images
# Skin mask of the training images
# Test set
Each training image and its mask have the same size. For each pixel in the original image there is a corresponding pixel in the skin mask with. The corresponding pixel in the mask is not skin the pixel has the RGB (1, 1, 1); otherwise the skin pixel is present.

## Training
### Inititalize
The SkinRecognition class has the following attributes:
* imgpath: Path to images (for training or testing).
* maskpath: Path to skin mask images.
* rect: The MxN neighborhood grid. 
  * default: (1, 1). This means only the pixel itself is considered as a feature.
* knn_neighborhoods: K Nearest Neighbors to be used in the KNN algorithm. 
  * default: 7
* img_paths: list of images provided by imgpath.
* mask_paths: list of masks provided by maskpath.
* X_training: Features of input images.
* Y_training: Labels of the features.
* knn: The KNeighborsClassifier object.
* confusion_matrix: If in test method generate_confusion_matrix is set, this attribute holds the average confusion matrix over all the tested images. Each confusion matrix is output to a csv file.
### Train
For this the method _get_training_images_ is used. The input parameters are listed below:
* use_kmeans_reduction: Whether to use kmeans reduction technique or not. This way after reading each set of image, the number of output features are reduced by a specified factor.
  * default: True
* skin_clusters: The K value of the Kmeans algorithm. The number of output centroids. if the value is between 0 and 1, the number of clusters would be calculated as `skin_clusters×length(features)`. It is ignored if `use_kmeans_reduction` is set to False.
* non_skin_clusters: The same logic as for skin_clusters applied to non_skin_clusters.
* knn_neighborhoods: K Nearest Neighbors to be used in the KNN algorithm
  * default: None, which uses the same value as provided in the initializer.
* img_start_index: The index where training would start on the list of images obtained from imgpath.
* img_count: Number of images to be read in the training.
* slice_size: How many images to be read in each iteration.
  * default: 3
  * This is specially useful in two scenarios:
    * When output model is set to a file so after each iteration over the slice the features are stored in the disk,
    * When using kmeans reduction, the numbers of skin and non-skin clusters are applied to the iteration
* output_model: The path and filename where the output model should be stored.
  * default: None. The output model will not be stored in disk.
* image_reduction_factor: The reduction factor to be applied on the image.
  * default: 0. No reduction is applied on the image.

## Testing
```python
sk = SkinRecognition(imgpath='PATH/TO/IMAGES', maskpath='PATH/TO/MASKS', rect=(7, 7), 
  knn_neighborhoods=8, model='PATH/TO/MODEL')
sk.test(img_test_count=10, img_count=0, method='all', generate_confusion_matrix=True, 
  output_results='PATH/TO/RESULTS', output_mask_path='PATH/TO/MASKS')
```
The parameters for this method are:
* test_count: The number of images from the list obtained from imgpath to be tested.
* test_start_index: The first image from the list obtained from imgpath should be the starting point.
* method: 'rect' or 'all'
  * default: 'rect': Only tests over the MxN grids. This method is faster but does not create neighborhoods for each pixel and predicts the class over each grid.
  * 'all': Creates MxN neighborhood for each pixel and predicts the class one by one.
* generate_confusion_matrix: Whether to cross validate the image if the mask for it exists. It iterates through each predicted mask and compares it against the existing mask.
  * default: True. It can be done for cross validation when masks exist.
* output_results: Path to output result file which is in CSV format. It includes the path to the file, the time needed for test, and the confusion matrix of the prediction.
  * default: None. Does not generate output results.
* output_mask_path: The path to store the masks.
  * default: None. Does not generate the predicted mask image file. Instead shows the results on screen using the matplotlib library.
