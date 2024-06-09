import osimport numpy as npimport torchimport pandas as pdfrom PIL import Imagefrom sklearn.metrics.pairwise import cosine_similarityfrom torchvision import models, transformsdef get_model(model_name):    if model_name == 'resnet18':        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)    elif model_name == 'resnet34':        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)    elif model_name == 'resnet50':        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)    elif model_name == 'resnet101':        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)    elif model_name == 'resnet152':        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)    elif model_name == 'vgg16':        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)    elif model_name == 'vgg19':        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)    elif model_name == 'inception_v3':        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)    elif model_name == 'densenet121':        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)    elif model_name == 'densenet169':        model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)    elif model_name == 'efficientnet_b0':        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)    elif model_name == 'efficientnet_b7':        model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)    else:        raise ValueError(f"Model {model_name} is not supported.")    model.eval()    return modelpreprocess = transforms.Compose([    transforms.Resize(256),    transforms.CenterCrop(224),    transforms.ToTensor(),    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])def extract_features(image_path, model_name):    """    Extract features from an image using a specified model.        Args:        image_path (str): Path to the image file.        model_name (str): Name of the model to use for feature extraction.        Returns:        np.ndarray: Extracted features as a flattened numpy array.    """    model = get_model(model_name)    input_image = Image.open(image_path).convert("RGB")    input_tensor = preprocess(input_image)    input_batch = input_tensor.unsqueeze(0)    with torch.no_grad():        output = model(input_batch)    return output.numpy().flatten()def is_in_domain(image_path, gallery_features, model_name, threshold):    """    Check if an image is in-domain based on cosine similarity with gallery features.        Args:        image_path (str): Path to the image file.        gallery_features (np.ndarray): Array of gallery features.        model_name (str): Name of the model to use for feature extraction.        threshold (float): Similarity threshold for determining in-domain status.        Returns:        bool: True if image is in-domain, False otherwise.    """    input_features = extract_features(image_path, model_name)    similarities = cosine_similarity([input_features], gallery_features)    max_similarity = np.max(similarities)    return max_similarity >= thresholddef evaluate_model(gallery_features, test_folders, model_name, threshold):    """    Evaluate the model on a set of test images from multiple folders.        Args:        gallery_features (np.ndarray): Array of gallery features.        test_folders (list): List of tuples containing (test_folder, true_label).        model_name (str): Name of the model to use for feature extraction.        threshold (float): Similarity threshold for determining in-domain status.        Returns:        dict: Dictionary with folder names as keys and (correct, total, accuracy) as values.    """    results = {}        for test_folder, true_label in test_folders:        correct = 0        total = 0        for root, _, files in os.walk(test_folder):            test_files = [f for f in files if f.endswith((".jpg", ".png", ".jpeg"))]            total += len(test_files)            for filename in test_files:                img_path = os.path.join(root, filename)                result = is_in_domain(img_path, gallery_features, model_name, threshold)                if result == true_label:                    correct += 1        accuracy = correct / total if total > 0 else 0        results[test_folder] = (correct, total, accuracy)        return resultsdef eval_image_folder(image_folder, gallery_features, model_name, threshold):    """    Evaluate all images in a folder, output the number of in-domain and out-of-domain images,    and create a CSV file with the results.        Args:        image_folder (str): Path to the folder containing images.        gallery_features (np.ndarray): Array of gallery features.        model_name (str): Name of the model to use for feature extraction.        threshold (float): Similarity threshold for determining in-domain status.        Returns:        None    """    in_domain_count = 0    out_of_domain_count = 0    results = []    for filename in os.listdir(image_folder):        if filename.endswith(".jpg") or filename.endswith(".png"):            img_path = os.path.join(image_folder, filename)            is_in = is_in_domain(img_path, gallery_features, model_name, threshold)            domain_status = "in-domain" if is_in else "out-of-domain"            results.append([filename, domain_status])            if is_in:                in_domain_count += 1            else:                out_of_domain_count += 1    # Create DataFrame and save to CSV    df = pd.DataFrame(results, columns=["Image", "Domain Status"])    df.to_csv("image_domain_evaluation.csv", index=False)        print(f"Total images: {len(results)}")    print(f"In-domain images: {in_domain_count}")    print(f"Out-of-domain images: {out_of_domain_count}")    print("Results saved to image_domain_evaluation.csv")