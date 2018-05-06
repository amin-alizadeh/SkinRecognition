import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from os import listdir, makedirs
from os.path import isfile, join, exists, basename, splitext
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from skimage import io, color
import csv

class SkinRecognition:
    imgpath = ""
    maskpath = ""
    rect = (1, 1)
    knn_neighborhoods = 7

    img_paths = []
    mask_paths = []

    X_training = 0
    Y_training = 0
    knn = None
    confusion_matrix = None

    def __init__(self, imgpath, maskpath, rect=(1, 1), knn_neighborhoods=7, model=None):
        self.imgpath = imgpath
        self.maskpath = maskpath
        self.rect = rect

        if model is not None:
            try:
                self.X_training = np.genfromtxt(model + '_X.csv', delimiter=';')
                self.Y_training = np.genfromtxt(model + '_Y.csv', delimiter=';')
                self.knn = KNeighborsClassifier(n_neighbors=knn_neighborhoods)
                self.knn.fit(self.X_training, self.Y_training)

            except IOError:
                print("File not found")
                raise

        self.img_paths = [join(imgpath, f) for f in listdir(imgpath) if
                          isfile(join(imgpath, f)) and (f[-4:] == ".jpg" or f[-4:] == ".bmp" or f[-5:] == ".jpeg")]
        self.img_paths.sort()
        self.mask_paths = [join(maskpath, f) for f in listdir(maskpath) if
                           isfile(join(maskpath, f)) and (f[-4:] == ".jpg" or f[-4:] == ".bmp" or f[-5:] == ".jpeg")]
        self.mask_paths.sort()

    def read_images(self, img_paths, mask_paths, image_reduction_factor=0):
        skin = []
        non_skin = []
        image_reduction_factor = np.int_(image_reduction_factor)
        for ind in range(len(img_paths)):
            try:
                img = misc.imread(img_paths[ind]) / 255
                msk = misc.imread(mask_paths[ind]) / 255
                if image_reduction_factor > 1:
                    img = self.reduce_image(img, image_reduction_factor)
                    msk = self.reduce_image(msk, image_reduction_factor)

                if np.array_equiv(img.shape, msk.shape):  # Ensures the image and mask are the same size
                    [x, y, z] = np.int_(np.floor([img.shape[0] / self.rect[0], img.shape[1] / self.rect[1],
                                                  img.shape[2] * (self.rect[0] * self.rect[1])]))

                    ignore_color = np.ones(z)
                    for i in range(x):
                        for j in range(y):
                            _img = img[i * self.rect[0]:(i + 1) * self.rect[0],
                                                   j * self.rect[1]:(j + 1) * self.rect[1], :].reshape(z)
                            _msk = msk[i * self.rect[0]:(i + 1) * self.rect[0],
                                                   j * self.rect[1]:(j + 1) * self.rect[1], :].reshape(z)

                            if np.array_equiv(_msk, ignore_color):
                                non_skin.append(_img)
                            else:
                                skin.append(_img)

            except IOError:
                pass
        return np.array(skin), np.array(non_skin)

    @staticmethod
    def get_number_clusters(n_clusters, n_points):
        km_clusters = n_clusters
        if n_clusters <= 0 or n_clusters > n_points:
            n_clusters = 0.1
        if n_clusters > 0 and n_clusters < 1:
            km_clusters = np.int_(n_clusters * n_points)
        return km_clusters

    def get_training_images(self, use_kmeans_reduction=True, skin_clusters=0.1, non_skin_clusters=0.1,
                            knn_neighborhoods=None, img_start_index=0, img_count=10, slice_size=3,
                            output_model=None, image_reduction_factor=0):

        if knn_neighborhoods is None:
            knn_neighborhoods = self.knn_neighborhoods

        img_paths = self.img_paths[img_start_index:img_start_index+img_count]
        mask_paths = self.mask_paths[img_start_index:img_start_index+img_count]

        self.X_training = np.zeros(self.rect[0] * self.rect[1] * 3)
        self.Y_training = np.int_(np.zeros(1))


        if output_model is not None:
            csvfile_X = open(output_model+'_X.csv', 'w', newline='')
            csvwriter_X = csv.writer(csvfile_X, delimiter=';')
            csvfile_Y = open(output_model + '_Y.csv', 'w', newline='')
            csvwriter_Y = csv.writer(csvfile_Y, delimiter=';')

        max_size = len(img_paths)

        iterations = np.int_(np.ceil(max_size / slice_size))
        for s in range(iterations):
            print((s + 1), "of", iterations, "has started...")
            slice_start = s * slice_size
            slice_end = (s + 1) * slice_size
            if slice_end > max_size:
                slice_end = max_size

            skin, non_skin = self.read_images(img_paths[slice_start:slice_end], mask_paths[slice_start:slice_end], image_reduction_factor)

            if use_kmeans_reduction:
                km_clusters = self.get_number_clusters(skin_clusters, skin.shape[0])
                kmeans_skin = KMeans(n_clusters=km_clusters, n_init=10, max_iter=20)
                kmeans_skin.fit(skin)

                km_clusters = self.get_number_clusters(non_skin_clusters, non_skin.shape[0])
                kmeans_non_skin = KMeans(n_clusters=km_clusters, n_init=10, max_iter=20)
                kmeans_non_skin.fit(non_skin)
                x_t = np.vstack((kmeans_skin.cluster_centers_, kmeans_non_skin.cluster_centers_))
                y_t = np.append(np.int_(np.ones(kmeans_skin.cluster_centers_.shape[0])),
                                np.int_(np.zeros(kmeans_non_skin.cluster_centers_.shape[0])))

            else:
                x_t = np.vstack((skin, non_skin))
                y_t = np.append(np.int_(np.ones(skin.shape[0])),
                                np.int_(np.zeros(non_skin.shape[0])))

            self.X_training = np.vstack((self.X_training, x_t))
            self.Y_training = np.append(self.Y_training, y_t)
            print((s+1), "out of", iterations, "iterations has been completed...")
            if output_model is not None:
                csvwriter_X.writerows(x_t)
                csvwriter_Y.writerows(y_t.reshape(-1, 1))
                csvfile_X.flush()
                csvfile_Y.flush()

        if output_model is not None:
            csvfile_X.close()
            csvfile_Y.close()

        self.knn = KNeighborsClassifier(n_neighbors=knn_neighborhoods)
        self.knn.fit(self.X_training, self.Y_training)

    """
    @method: 'rect' or 'all'
    """

    def test(self, test_count, test_start_index, method='rect', generate_confusion_matrix=True, output_results=None,
             output_mask_path=None):
        method_rect = True
        if method == 'all':
            method_rect = False

        if output_mask_path is None:
            columns = 2
            rows = np.int_(np.ceil(test_count / columns))
            plt.clf()
            fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(12, test_count * 2))
            if columns * rows > 1:
                axes = axes.flatten()
        else:
            if not exists(output_mask_path):
                makedirs(output_mask_path)

        all_tp = 0
        all_tn = 0
        all_fp = 0
        all_fn = 0

        if output_results is not None:
            csvfile_cm = open(output_results + '_' + method + '.csv', 'w', newline='')
            csvwriter_cm = csv.writer(csvfile_cm, delimiter=';')

        for ind in range(test_start_index, test_start_index + test_count):
            img = misc.imread(self.img_paths[ind]) / 255
            print("Processing image....", ind)

            start_time = time.time()

            if output_results is not None:
                csvwriter_cm.writerows([[start_time, ind]])
                csvfile_cm.flush()

            if method_rect:
                _img, [x, y, z] = self.test_rect_pixels(img)
            else:
                _img, [x, y, z] = self.test_all_pxiels(img)

            end_time = time.time()
            elapsed_time = np.round(end_time - start_time, 3)
            tm = "{:d} minutes and {:.3f} seconds".format(np.int_(elapsed_time / 60), elapsed_time % 60)

            if output_results is not None:
                csvwriter_cm.writerows([[tm, self.img_paths[ind]]])
                csvfile_cm.flush()

            if generate_confusion_matrix:
                msk = misc.imread(self.mask_paths[ind])[:self.rect[0] * x, :self.rect[1] * y, :] / 255
                tp = 0
                tn = 0
                fp = 0
                fn = 0
                white = np.ones(3)
                for x1 in range(_img.shape[0]):
                    for y1 in range(_img.shape[1]):
                        img_is_white = np.array_equiv(_img[x1, y1, :], white)
                        msk_is_white = np.array_equiv(msk[x1, y1, :], white)
                        if (not img_is_white) and (not msk_is_white):
                            tp += 1
                        elif img_is_white and msk_is_white:
                            tn += 1
                        elif img_is_white and not msk_is_white:
                            fn += 1
                        else:
                            fp += 1
                pixel_count = _img.shape[0] * _img.shape[1]

                all_tp += np.round(tp / pixel_count, 5)
                all_tn += np.round(tn / pixel_count, 5)
                all_fp += np.round(fp / pixel_count, 5)
                all_fn += np.round(fn / pixel_count, 5)

                if output_results is not None:
                    csvwriter_cm.writerows([[tp, tn], [fp, fn]])
                    csvfile_cm.flush()

            print("Processed image", ind)

            if output_mask_path is None:
                axes[ind - test_start_index].imshow(_img)
            else:
                mask_file_name = output_mask_path + '/' + splitext(basename(self.img_paths[ind]))[
                    0] + "_" + method + ".jpg"
                misc.imsave(mask_file_name, _img)

        if output_results is not None:
            csvfile_cm.close()

        if output_mask_path is None:
            plt.show()

        if generate_confusion_matrix:
            self.confusion_matrix = np.array(
                [[all_tp / test_count, all_tn / test_count], [all_fp / test_count, all_fn / test_count]])
        else:
            self.confusion_matrix = None

    def test_rect_pixels(self, img):
        [x, y, z] = np.int_(np.floor(
            [img.shape[0] / self.rect[0], img.shape[1] / self.rect[1], img.shape[2] * (self.rect[0] * self.rect[1])]))
        img = img[:self.rect[0] * x, :self.rect[1] * y, :]
        X_test = np.empty((x * y, z))
        for i in range(x):
            for j in range(y):
                X_test[(i * y) + j, :] = img[i * self.rect[0]:(i + 1) * self.rect[0],
                                         j * self.rect[1]:(j + 1) * self.rect[1], :].reshape(z)

        pred = self.knn.predict(X_test).reshape(x, y, 1)

        _pred = np.empty((img.shape[0], img.shape[1], 1))
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                _pred[i * self.rect[0]:(i + 1) * self.rect[0], j * self.rect[1]:(j + 1) * self.rect[1], :] = \
                    pred[i, j, 0] * np.ones((self.rect[0], self.rect[1], 1))

        _img = np.multiply(_pred, img)
        img_mask = np.zeros((_img.shape[0], _img.shape[1], _img.shape[2]))
        _img[np.where(_img == img_mask)] = 1
        return _img, [x, y, z]

    def test_all_pxiels(self, img):
        [x, y, z] = np.int_(np.floor(
            [img.shape[0] / self.rect[0], img.shape[1] / self.rect[1], img.shape[2] * (self.rect[0] * self.rect[1])]))

        if img.shape[0] < self.rect[0]:
            self.rect[0] = img.shape[0]
        if img.shape[1] < self.rect[1]:
            self.rect[1] = img.shape[1]

        rx_before = np.int_(np.ceil(self.rect[0] / 2)) - 1
        ry_before = np.int_(np.ceil(self.rect[1] / 2)) - 1
        rx_after = np.int_(self.rect[0] / 2)
        ry_after = np.int_(self.rect[1] / 2)

        shp = img.shape[2] * self.rect[0] * self.rect[1]

        im = np.ones((img.shape[0] - rx_before - rx_after, img.shape[1] - ry_before - ry_after,
                      img.shape[2] * self.rect[0] * self.rect[1]))

        for a in range(rx_before, img.shape[0] - rx_after):
            for b in range(ry_before, img.shape[1] - ry_after):
                _im = img[a - rx_before:a - rx_before + self.rect[0], b - ry_before:b - ry_before + self.rect[1]]
                im[a - rx_before, b - ry_before, :] = _im.reshape(shp)

        pred = self.knn.predict(im.reshape(im.shape[0] * im.shape[1], im.shape[2]))

        img_r = img[rx_before:img.shape[0] - rx_after, ry_before: img.shape[1] - ry_after, :]
        _img = np.multiply(pred.reshape(img_r.shape[0], img_r.shape[1], 1), img_r)
        img_mask = np.zeros((_img.shape[0], _img.shape[1], _img.shape[2]))
        _img[np.where(_img == img_mask)] = 1
        return _img, [x, y, z]

    """
    @color_mode: rgb, hsv, lab
    """

    def reduce_image(self, image, n_pixels=2, color_mode='rgb', return_color_mode=None):
        if color_mode == 'rgb':
            image = color.rgb2lab(image)
        elif color_mode == 'hsv':
            image = color.hsv2lab(image)
        elif color_mode == 'lab':
            pass
        else:
            raise ValueError('color mode is undefined. It must be one of "rgb", "hsv" or "lab".')
            return
        c1 = 0
        c2 = 0
        _d1, _d2 = np.int_(np.ceil(image.shape[0] / (2 * n_pixels + 1))), \
                   np.int_(np.ceil(image.shape[1] / (2 * n_pixels + 1)))
        _image = np.ones((_d1, _d2, 3))  # * 2
        for i in range(n_pixels, image.shape[0], n_pixels * 2 + 1):
            c2 = 0
            for j in range(n_pixels, image.shape[1], n_pixels * 2 + 1):
                for k in range(3):
                    _image[c1, c2, k] = image[i - n_pixels:i + n_pixels + 1, j - n_pixels:j + n_pixels + 1, k].mean()
                c2 += 1
            c1 += 1

        if return_color_mode == None:
            return_color_mode = color_mode

        if return_color_mode == 'rgb':
            _image = color.lab2rgb(_image)
            # Since not all the columns and rows are covered in the image reduction, we make sure that a "white" pixel is placed in the missing place
            _image[c1:_d1, :, :] = np.ones((1, _d2, 3))
            _image[:, c2:_d2, :] = np.ones((_d1, 1, 3))
        elif return_color_mode == 'hsv':
            _image = color.lab2hsv(_image)
        return _image