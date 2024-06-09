import torch
import numpy as np
import pandas as pd
from out_of_domain.similarity_check import is_in_domain, evaluate_model, eval_image_folder

def test_similarity_check():
    test_image_folder = "test_images"
    model_name = "resnet50"
    threshold = 0.85
    output_file = f"{model_name}.npy"
    
   
    
    gallery_features = np.load(output_file)
    
   
    print("Checking if a single image is in-domain...")
    test_image_path = "/Users/surajgautam/out_of_domain_library/test_images/image_0.jpg"
    result = is_in_domain(test_image_path, gallery_features, model_name, threshold)
    print(f"Test image is {'in-domain' if result else 'out-of-domain'}")
    
    # Evaluate the model on multiple folders
    test_folder_in_domain = "test_folder_in_domain"
    test_folder_out_of_domain = "test_folder_out_of_domain"
    
    
    test_folders = [
        (test_folder_in_domain, True),
        (test_folder_out_of_domain, False)
    ]
    
    # Evaluate the model
    print("Evaluating the model on multiple folders...")
    results = evaluate_model(gallery_features, test_folders, model_name, threshold)
    for folder, (correct, total, accuracy) in results.items():
        print(f"Folder: {folder}, Correct: {correct}, Total: {total}, Accuracy: {accuracy:.4f}")
    
    # Evaluate the image folder
    print("Evaluating the image folder...")
    eval_image_folder(test_image_folder, gallery_features, model_name, threshold)
    
    # Load and print the results from the CSV file
    results_df = pd.read_csv("image_domain_evaluation.csv")
    print("Results from image_domain_evaluation.csv:")
    print(results_df)
    
  

# Run the test function
if __name__ == "__main__":
    test_similarity_check()
