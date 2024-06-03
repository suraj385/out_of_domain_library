import numpy as npfrom out_of_domain.feature_extraction import extract_featuresfrom out_of_domain.similarity_check import is_in_domain, evaluate_model# Dummy data for testingdummy_image_path = "dummy_image.jpg"dummy_gallery_features = np.random.rand(5, 2048)  # Assuming 2048 features from ResNet50# Create a dummy image for testingfrom PIL import Imageimg = Image.new('RGB', (224, 224), color=(73, 109, 137))img.save(dummy_image_path)def test_extract_features():    features = extract_features(dummy_image_path)    assert features.shape == (2048,), "Feature extraction failed"def test_is_in_domain():    result = is_in_domain(dummy_image_path, dummy_gallery_features)    assert isinstance(result, bool), "In-domain check failed"def test_evaluate_model():    correct, total, accuracy = evaluate_model(dummy_gallery_features, ".", True)    assert isinstance(correct, int) and isinstance(total, int) and isinstance(accuracy, float), "Model evaluation failed"def run_tests():    test_extract_features()    test_is_in_domain()    test_evaluate_model()    print("All tests passed.")if __name__ == "__main__":    run_tests()# Clean up dummy imageimport osos.remove(dummy_image_path)