"""
https://towardsdatascience.com/dimensionality-reduction-of-a-color-photo-splitting-into-rgb-channels-using-pca-algorithm-in-python-ba01580a1118
https://towardsdatascience.com/pca-clearly-explained-how-when-why-to-use-it-and-feature-importance-a-guide-in-python-7c274582c37e
https://towardsdatascience.com/feature-extraction-using-principal-component-analysis-a-simplified-visual-demo-e5592ced100a
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
from scipy.stats import stats
import matplotlib.image as mpimg

if __name__ == "__main__":
    img = cv2.imread('/Users/sendo_mac/Documents/avp/Machine-Learning/PCA/data/images.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    blue, green, red = cv2.split(img)
    #
    # fig = plt.figure(figsize=(15, 7.2))
    # fig.add_subplot(131)
    # plt.title("Blue Channel")
    # plt.imshow(blue)
    #
    # fig.add_subplot(132)
    # plt.title("Green Channel")
    # plt.imshow(green)
    #
    # fig.add_subplot(133)
    # plt.title("Red Channel")
    # plt.imshow(red)
    #
    # plt.show()

    # blue_temp_df = pd.DataFrame(data=blue)

    df_blue = blue / 255
    df_green = green / 255
    df_red = red / 255

    print(df_blue.shape)
    print(df_green.shape)
    print(df_red.shape)

    pca_b = PCA(n_components=50)
    pca_b.fit(df_blue)
    trans_pca_b = pca_b.transform(df_blue)

    pca_g = PCA(n_components=50)
    pca_g.fit(df_green)
    trans_pca_g = pca_g.transform(df_green)

    pca_r = PCA(n_components=50)
    pca_r.fit(df_red)
    trans_pca_r = pca_r.transform(df_red)

    print(trans_pca_b.shape)
    print(trans_pca_g.shape)
    print(trans_pca_r.shape)

    print(f"Blue Channel : {sum(pca_b.explained_variance_ratio_)}")
    print(f"Green Channel: {sum(pca_g.explained_variance_ratio_)}")
    print(f"Red Channel  : {sum(pca_r.explained_variance_ratio_)}")

    """
    data_original = np.dot(data_reduced, pca.components_) + pca.mean_
    """

    b_arr = pca_b.inverse_transform(trans_pca_b)
    g_arr = pca_g.inverse_transform(trans_pca_g)
    r_arr = pca_r.inverse_transform(trans_pca_r)
    print(b_arr.shape, g_arr.shape, r_arr.shape)

    img_reduced = (cv2.merge((b_arr, g_arr, r_arr)))
    print(img_reduced.shape)

    fig = plt.figure(figsize=(10, 7.2))
    fig.add_subplot(121)
    plt.title("Original Image")
    plt.imshow(img)
    fig.add_subplot(122)
    plt.title("Reduced Image")
    plt.imshow(img_reduced)
    plt.show()
