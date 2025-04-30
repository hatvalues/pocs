from pinterest.image_downloader import PinterestBoardDownloader
from selenium import webdriver
import pytest
import os
from pathlib import Path
import imghdr


@pytest.fixture
def pinterest_downloader(tmp_path):
    output_dir = os.path.join(tmp_path, "images_download")
    downloader = PinterestBoardDownloader(
        board_url="https://de.pinterest.com/julianhatwell/discrae/by-me/",
        output_dir=output_dir,
    )
    yield downloader
    if downloader.driver:
        downloader.close_driver()


def test_setup_driver(pinterest_downloader):
    """Test the setup_driver method"""
    pinterest_downloader.setup_driver()
    assert hasattr(pinterest_downloader, "driver")
    assert isinstance(pinterest_downloader.driver, webdriver.Chrome)
    assert pinterest_downloader.board_name == "julianhatwell_discrae_byme"


def test_extract_file_name(pinterest_downloader):
    with pytest.raises(ValueError):
        pinterest_downloader.extract_file_name("dummy")
    assert (
        pinterest_downloader.extract_file_name(
            "https://de.pinterest.com/pin/397513104614841320/"
        )
        == "397513104614841320"
    )
    assert (
        pinterest_downloader.extract_file_name(
            "https://i.pinimg.com/1200x/88/3c/ab/883cab882a4b67bc2aa05555c5ece333.jpg"
        )
        == "883cab882a4b67bc2aa05555c5ece333.jpg"
    )


# Commented out. Don't wanna hit Pinterest for every test run.
# def test_download_image(pinterest_downloader):
#     with pytest.raises(ValueError):
#         pinterest_downloader.download_image(image_url="dummy")

#     pinterest_downloader.download_image(image_url="https://de.pinterest.com/pin/397513104614841320/")
#     file_path = Path(pinterest_downloader.output_dir, "397513104614841320")
#     assert file_path.exists()

#     pinterest_downloader.download_image(image_url="https://i.pinimg.com/1200x/88/3c/ab/883cab882a4b67bc2aa05555c5ece333.jpg")
#     file_path = Path(pinterest_downloader.output_dir, "883cab882a4b67bc2aa05555c5ece333.jpg")
#     assert file_path.exists()
#     assert imghdr.what(file_path) == "jpeg"
