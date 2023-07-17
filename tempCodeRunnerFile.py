xtest_image = image.load_img(img_path, target_size=(224, 224))
    xtest_image = image.img_to_array(xtest_image)
    xtest_image = np.expand_dims(xtest_image, axis = 0)
    preds = model.predict_classes(xtest_image)
    return preds
