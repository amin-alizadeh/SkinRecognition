import time
from skin_recgnition import SkinRecognition


def main():
    #Path to images, training/test
    imgpath = 'datasets/training'
    #Path to masks. Used for training or corss validation
    maskpath = 'datasets/skin'
    #Value for K Nearest Neighbors
    neighborhoods = 5
    #MxN neighborhoods
    rect = (7, 7)
    #Starting index for training
    img_start_index = 0
    #Number of umages to be used in training
    img_count = 10
    #Number of images to be read at once. KMeans reduction is also applied over a slice
    slice_size = 5
    use_kmeans_reduction = True
    #KMeans reduction factor for skin clusters
    skin_clusters = 0.07
    # KMeans reduction factor for non-skin clusters
    non_skin_clusters = 0.07
    #Path to where the generated model should be stored.
    model_path = 'models'

    #Use the next 2 lines to only create training model
    sk = SkinRecognition(imgpath, maskpath, rect, knn_neighborhoods=neighborhoods)
    sk.get_training_images(use_kmeans_reduction, skin_clusters, non_skin_clusters, neighborhoods,
                           img_start_index, img_count, slice_size, model_path)

    #Use the next line to load an existing training model
    #sk = SkinRecognition(imgpath, maskpath, rect, knn_neighborhoods=neighborhoods, model=model_path)

    #Starting index of the images where testing begins
    start_image_index = 0
    #Number of images to be tested
    img_test_count = 2
    #The method for testing. all for each pixel, rect for MxN neighborhood grids
    method = 'all'
    #Path to store the test results
    res = 'tmp/results/cm_' + str(start_image_index) + "_" + str(img_test_count)
    #Path to store the masks. If None the results are plotted
    output_mask_path = 'tmp/results/masks'

    sk.test(img_test_count, start_image_index, method=method, generate_confusion_matrix=True, output_results=res, output_mask_path=output_mask_path)

if __name__ == '__main__':
    main()

