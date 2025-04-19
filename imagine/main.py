from src.imagine import datasets as ds

dataset = ds.CustomImageDataset(image_directory="images")

if __name__ == "__main__":
    print("Dataset initialized successfully.")
    print(f"Number of images: {len(dataset.image_paths)}")
    print(f"First image path: {dataset.image_paths[0]}")
    # Add any additional test cases or functionality you want to check