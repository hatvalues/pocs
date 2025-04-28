from pinterest.image_downloader import PinterestBoardDownloader
from selenium import webdriver
import pytest
import os

@pytest.fixture
def pinterest_downloader(tmp_path):
    output_dir = os.path.join(tmp_path, "images_download")
    downloader = PinterestBoardDownloader(output_dir)
    yield downloader
    if downloader.driver:
        downloader.close_driver()

def test_setup_driver(pinterest_downloader):
    """Test the setup_driver method"""
    pinterest_downloader.setup_driver()
    assert hasattr(pinterest_downloader, "driver")
    assert isinstance(pinterest_downloader.driver, webdriver.Chrome)