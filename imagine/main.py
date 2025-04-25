# from src.imagine import datasets as ds
from pinterest.image_downloader import PinterestBoardDownloader

# dataset = ds.CustomImageDataset(image_directory="images")

# if __name__ == "__main__":
#     print("Dataset initialized successfully.")
#     print(f"Number of images: {len(dataset.image_paths)}")
#     print(f"First image path: {dataset.image_paths[0]}")
#     # Add any additional test cases or functionality you want to check


def main():
    # Ask for Pinterest board URL or use the default
    default_url = "https://www.pinterest.com/julianhatwell/discrae/by-me/"
    board_url = input(
        f"Enter Pinterest board URL (press Enter for default: {default_url}): "
    )
    if not board_url:
        board_url = default_url

    # Ask if login is required
    # login_required = input("Do you need to log in to access this board? (y/n): ").lower() == 'y'
    # username = None
    # password = None

    # if login_required:
    #     username = input("Enter Pinterest username or email: ")
    #     password = input("Enter Pinterest password: ")

    # Create downloader and download the board
    downloader = PinterestBoardDownloader()
    downloader.download_board(board_url)


if __name__ == "__main__":
    main()
