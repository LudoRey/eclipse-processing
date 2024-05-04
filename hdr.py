import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import interpolate

def saturation_weighting(img, low, high, low_smoothness, high_smoothness):
    low_weights = np.clip((img + low_smoothness - low) / low_smoothness, 0, 1)
    high_weights = np.clip((- img + high_smoothness + high) / high_smoothness, 0, 1)
    weights = low_weights + high_weights - 1 
    return weights

def binary_disk(x_c, y_c, radius, shape):
    x, y = np.arange(shape[1]), np.arange(shape[0])
    X, Y = np.meshgrid(x, y)
    binary_disk = np.sqrt((X-x_c)**2 + (Y-y_c)**2) <= radius
    return binary_disk

def angle_map(x_c, y_c, shape):
    x, y = np.arange(shape[1]), np.arange(shape[0])
    X, Y = np.meshgrid(x, y)
    theta = np.arctan(-(Y-y_c)/(X-x_c)) + np.pi * (X-x_c < 0) + 2*np.pi * (X-x_c > 0)*(Y-y_c > 0) # tan is pi-periodic : arctan can be many things
    return theta

def evaluate_trigonometric_basis(theta, degree):
    out = [np.ones(theta.shape[0])]
    for n in range(1, degree+1):
        out.append(np.cos(n*theta))
        out.append(np.sin(n*theta))
    out = np.stack(out, axis=1) 
    return out

def resample_per_sector(theta, num_sectors, num_samples_per_sector):
    resampled_indices_per_sector = np.zeros([num_sectors, num_samples_per_sector], dtype=np.uint64)
    for sector_idx in range(num_sectors):
        # Define sector mask
        theta_min, theta_max = 2*np.pi*sector_idx/num_sectors, 2*np.pi*(sector_idx+1)/num_sectors
        sector_mask = (theta >= theta_min)*(theta < theta_max)
        # Resample sector indices
        sector_indices = np.nonzero(sector_mask)[0]
        quotient, remainder = np.divmod(num_samples_per_sector, len(sector_indices))
        resampled_indices = np.concatenate([np.tile(sector_indices, quotient), np.random.choice(sector_indices, size=remainder, replace=False)])
        resampled_indices_per_sector[sector_idx] = resampled_indices
    return resampled_indices_per_sector

def linear_trigo_fit(x, theta, y, degree):
    # Construct trigonometric polynomial features (example of a term : x*sin(theta))
    trigo_basis = evaluate_trigonometric_basis(theta, degree)
    X = np.concatenate([trigo_basis, trigo_basis*x[:,None]], axis=1)

    # Add penalty to high order terms
    # num_features = 2*(2*degree+1)
    # penalty_matrix = reg_factor*np.tile(np.arange(2*degree+1), 2)*np.eye(num_features)
    # X = np.concatenate([X, penalty_matrix], axis=0)
    # y = np.concatenate([y, np.zeros(num_features)], axis=0)

    reg = LinearRegression(fit_intercept=False).fit(X,y)

    offset_trigo_coeffs = reg.coef_[:1+2*degree]
    slope_trigo_coeffs = reg.coef_[1+2*degree:]
    return offset_trigo_coeffs, slope_trigo_coeffs

def fast_evaluate_trigonometric_basis(theta, degree, N=10000):
    # evaluate on uniformly spaced samples and use nearest interpolation
    linspace_theta = np.linspace(0, 2*np.pi, N)
    linspace_trigo_basis = evaluate_trigonometric_basis(linspace_theta, degree)
    f = interpolate.interp1d(linspace_theta, linspace_trigo_basis, kind='nearest', axis=0)
    trigo_basis = f(theta)
    return trigo_basis

def masked_resampled_linear_trigo_fit_images(img_x, img_theta, img_y, mask, degree=4, num_sectors=30, num_samples_per_sector=400):
    valid_x = img_x.mean(axis=2)[mask]
    valid_y = img_y.mean(axis=2)[mask]
    valid_theta = img_theta[mask]

    # Select fixed amount of samples per sector (so that we sample ~uniformly over theta)
    resampled_indices_per_sector = resample_per_sector(valid_theta, num_sectors, num_samples_per_sector)
    samples_per_sector_x, samples_per_sector_theta, samples_per_sector_y = valid_x[resampled_indices_per_sector], valid_theta[resampled_indices_per_sector], valid_y[resampled_indices_per_sector]

    # Fit all sectors at once using trigonometric interactions
    offset_trigo_coeffs, slope_trigo_coeffs = linear_trigo_fit(samples_per_sector_x.reshape(-1), 
                                                                samples_per_sector_theta.reshape(-1), 
                                                                samples_per_sector_y.reshape(-1), 
                                                                degree)


    #trigo_basis = evaluate_trigonometric_basis(img_theta.reshape(-1), degree) # really slow
    trigo_basis = fast_evaluate_trigonometric_basis(img_theta.reshape(-1), degree) # twice faster with default params

    img_offset = (trigo_basis @ offset_trigo_coeffs).reshape(img_theta.shape)
    img_slope = (trigo_basis @ slope_trigo_coeffs).reshape(img_theta.shape)
    img_fitted_x = img_offset[:,:,None] + img_slope[:,:,None]*img_x

    return img_fitted_x