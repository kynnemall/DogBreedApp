import copy
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from skimage.segmentation import quickshift


def run_lime(input_image, model, num_perturb=150, num_top_features=4):
    # taken from https://nbviewer.jupyter.org/url/arteagac.github.io/blog/lime_image.ipynb

    # calculate superpixels in the image
    superpixels = quickshift(input_image, kernel_size=4,max_dist=200, ratio=0.2)
    num_superpixels = np.unique(superpixels).shape[0]

    # perturb the image
    perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))

    # run predictions on images with perturbations
    predictions = []
    for pert in perturbations:
        perturbed_img = perturb_image(input_image, pert, superpixels)
        pred = model.predict(perturbed_img[np.newaxis,:,:,:])
        predictions.append(pred)
    predictions = np.array(predictions)
    
    # perturbation with all superpixels enabled 
    original_image = np.ones(num_superpixels)[np.newaxis,:]
    distances = pairwise_distances(perturbations,original_image, metric='cosine').ravel()
    kernel_width = 0.25
    weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function

    # compute linear model using weights and perturbations
    class_to_explain = top_pred_classes[0]
    simpler_model = LinearRegression()
    simpler_model.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
    coeff = simpler_model.coef_[0]
    top_features = np.argsort(coeff)[-num_top_features:]

    mask = np.zeros(num_superpixels) 
    mask[top_features]= True # activate top superpixels
    
    top_superpixels = perturb_image(input_image/2+0.5,mask,superpixels)
    return top_superpixels

def perturb_image(img, perturbation, segments):
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1 
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image*mask[:,:,np.newaxis]
    return perturbed_image
