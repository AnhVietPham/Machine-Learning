import numpy as np
import cv2
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from scipy import ndimage as nd
from skimage.filters import roberts, sobel, scharr, prewitt


def feature_extraction(img):
    df = pd.DataFrame()
    img2 = img.reshape(-1)
    df['Pixel_value'] = img2

    # Gabor features
    num = 1
    kernels = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
                    gabor_label = 'Gabor' + str(num)
                    ksize = 9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img
                    print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1

    # CANNY EDGE
    edges = cv2.Canny(img, 100, 200)
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1

    # Roberts edge
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1

    # Sobel
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1

    # Scharr
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1

    # Prewitt
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1

    # Gaussian with sigma = 3
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

    # Gaussian with sigma = 7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3

    # Median with sigma = 3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1

    # Variance with size = 3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img1
    return df


if __name__ == "__main__":
    filename = "model/sandstone_model"
    loaded_model = pickle.load(open(filename, 'rb'))
    image_path = "dataset/train_images/Sandstone_Versa0200.tif"
    img_read = cv2.imread(image_path)
    img = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
    X = feature_extraction(img)
    result = loaded_model.predict(X)
    segmented = result.reshape((img.shape))

    plt.imsave("predict/segmented.jpg", segmented, cmap='jet')
