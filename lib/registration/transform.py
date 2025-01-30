import skimage as sk

def centered_rigid_transform(center, rotation, translation):
    '''Rotate first around center, then translate'''
    t_uncenter = sk.transform.AffineTransform(translation=center)
    t_center = t_uncenter.inverse
    t_rotate = sk.transform.AffineTransform(rotation=rotation)
    t_translate = sk.transform.AffineTransform(translation=translation)
    return t_center + t_rotate + t_uncenter + t_translate

def rotate_translate(img, theta, dx, dy):
    print(f"Rotating image by :")
    print(f"    - theta = {theta:.4f}")
    print(f"Translating image by :")
    print(f"    - dx = {dx:.2f}")
    print(f"    - dy = {dy:.2f}")
    tform = centered_rigid_transform(center=(img.shape[1]/2, img.shape[0]/2), rotation=theta, translation=(dx, dy))
    registered_img = sk.transform.warp(img, tform.inverse)
    return registered_img
