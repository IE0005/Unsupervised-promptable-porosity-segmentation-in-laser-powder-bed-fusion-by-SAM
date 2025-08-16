import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io,filters
import pandas as pd
from sklearn.metrics import pairwise_distances
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from sklearn_extra.cluster import KMedoids
from sklearn.covariance import LedoitWolf
from PIL import Image
#For example of Sample 5, this can be used for all the other samples. 
image = cv2.imread('.../set1sample5raw_0000.tif', cv2.IMREAD_COLOR)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_height, image_width, _ = image.shape
# Set the path to your dataset folder
data_folder = '.../sample5'

# Load your images and convert them to a feature matrix
def load_images(folder, num_images=None):
    images = []
    count = 0

    for filename in os.listdir(folder):
        if num_images is not None and count >= num_images:
            break
        print(filename)
        img = Image.open(os.path.join(folder, filename))
        img_array = np.array(img)
        images.append(img_array.flatten())
        count += 1

    return np.array(images)

# Load the first 50 images and flatten them into feature vectors
num_images_to_load = 50
data = load_images(data_folder, num_images=num_images_to_load)
print("The data is loaded")

# Assuming you have a distance function defined
def mahalanobis_distance(x, y, precision_matrix):
    diff = x - y
    return np.sqrt(np.dot(np.dot(diff, precision_matrix), diff.T))

# Compute the covariance matrix using Ledoit-Wolf estimator
covariance_matrix = LedoitWolf().fit(data).precision_
print("The covariance matrix is done")

# Adjust the mahalanobis_distance function with the precision matrix
mahalanobis_distance_with_precision = lambda x, y: mahalanobis_distance(x, y, covariance_matrix)

# Optimal number of clusters
optimal_clusters = 3

# Perform KMedoids clustering with the optimal number of clusters and custom distance
kmedoids = KMedoids(n_clusters=optimal_clusters, metric=mahalanobis_distance_with_precision, random_state=0)
cluster_labels = kmedoids.fit_predict(data)
print("the kmedoids lables are done")

# Output folder for clustered images
output_folder = '.../k_meds/Clustered_tiff_images'
# Create an empty dictionary to store the mapping of filenames to cluster labels
filename_to_cluster = {}

# After performing K-medoids clustering, you will have 'cluster_labels' containing the cluster labels
# Iterate through the data points and their cluster labels
for i, cluster_label in enumerate(cluster_labels):
    # Retrieve the original filename for this data point
    original_filename = os.listdir(data_folder)[i]  # Assuming the list of filenames is in the same order as the data
    # print(original_filename)

    # Store the filename as the key and the cluster label as the value in the dictionary
    filename_to_cluster[original_filename] = cluster_label

# Display medoid images for each cluster
medoids_indices = kmedoids.medoid_indices_
medoid_images = data[medoids_indices]

# Loop through the data points and save the images in their respective cluster folders
for idx, original_filename in enumerate(filename_to_cluster):
    cluster_num = filename_to_cluster[original_filename]
    cluster_folder = os.path.join(output_folder, f'cluster_{cluster_num+1}')
    image_name = original_filename
    image_path = os.path.join(cluster_folder, image_name)

    # Create the directory if it doesn't exist
    os.makedirs(cluster_folder, exist_ok=True)

    # Assuming data[idx] is a flattened image data
    image_reshaped = np.reshape(data[idx], (image_height, image_width, 3)).astype(np.uint8)

    # Convert image data to PIL Image and save
    Image.fromarray(image_reshaped).save(image_path)

print("Images assigned to clusters and saved successfully.")
folder_path = '.../k_meds/medoids'  # Change this path as needed
os.makedirs(folder_path, exist_ok=True)
for i, centroid_image in enumerate(medoid_images):
    print(centroid_image.shape)
    centroid_image_reshaped = centroid_image.reshape((image_height,image_width, 3))
    print(centroid_image_reshaped.shape)
    print(type(centroid_image_reshaped))


    image_filename = os.path.join(folder_path, f'medoids_cluster_{i+1}.png')
    #print(image_filename)
    # print(centroid_image[0])
    Image.fromarray((centroid_image_reshaped).astype(np.uint8)).save(image_filename)
    #plt.savefig(image_filename, dpi=500)  # Save with the same DPI as the figure
    # plt.close()

print("Medoids images saved successfully.")
# Path to the folder containing your PNG images
image_folder = '.../k_meds/medoids'

# List all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.png')]

# Loop through each image
for image_file in image_files:
    # Read the image
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    print(image.shape)

    # Reshape the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image.reshape((-1, 3))  # 3 for RGB channels

    # Convert to float type
    pixel_vals = np.float32(pixel_vals)
    # Calculate the mean of pixel values for thresholding
    mean_pixel_value = np.mean(pixel_vals)
    print(mean_pixel_value)

    # Apply binary thresholding
    _, thresholded_image = cv2.threshold(image, mean_pixel_value, 255, cv2.THRESH_TOZERO_INV)
    mask = (thresholded_image > np.min(thresholded_image))
    #mask = th5 > 109
    # Set all corresponding pixels in the thresholded image to 255
    thresholded_image[mask] = 255
    print(np.unique(thresholded_image))
    print(thresholded_image.shape)

    # Convert thresholded_image to grayscale
    thresholded_image_gray = cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2GRAY)
    print(thresholded_image_gray.shape)


    # Output folder for thresholded images
    output_folder = '..../k_meds/med_centers'
    os.makedirs(output_folder, exist_ok=True)

    # Save the thresholded image
    output_path = os.path.join(output_folder, f'thresholded_{image_file}')
    cv2.imwrite(output_path, thresholded_image_gray)
    plt.subplot(1, 2, 2)
    plt.imshow(thresholded_image_gray, cmap='gray')
    plt.title('Thresholded Image')
    plt.axis("off")

    plt.show()