from pinterest.image_downloader import PinterestBoardDownloader
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.remote.webelement import WebElement
import pytest
import os
from pathlib import Path
from PIL import Image


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


def test_file_name_from_url(pinterest_downloader):
    with pytest.raises(ValueError):
        pinterest_downloader.file_name_from_url("dummy")
    assert (
        pinterest_downloader.file_name_from_url(
            "https://de.pinterest.com/pin/397513104614841320/"
        )
        == "397513104614841320"
    )
    assert (
        pinterest_downloader.file_name_from_url(
            "https://i.pinimg.com/1200x/88/3c/ab/883cab882a4b67bc2aa05555c5ece333.jpg"
        )
        == "883cab882a4b67bc2aa05555c5ece333.jpg"
    )


def test_image_file_parse(pinterest_downloader):
    assert (
        pinterest_downloader.convert_to_1200x(
            "https://i.pinimg.com/1200x/88/3c/ab/883cab882a4b67bc2aa05555c5ece333.jpg"
        )
        == "https://i.pinimg.com/1200x/88/3c/ab/883cab882a4b67bc2aa05555c5ece333.jpg"
    )
    assert (
        pinterest_downloader.convert_to_1200x(
            "https://i.pinimg.com/753x/88/3c/ab/883cab882a4b67bc2aa05555c5ece333.jpg"
        )
        == "https://i.pinimg.com/1200x/88/3c/ab/883cab882a4b67bc2aa05555c5ece333.jpg"
    )
    assert (
        pinterest_downloader.convert_to_1200x(
            "https://i.pinimg.com/236x/88/3c/ab/883cab882a4b67bc2aa05555c5ece333.jpg"
        )
        == "https://i.pinimg.com/1200x/88/3c/ab/883cab882a4b67bc2aa05555c5ece333.jpg"
    )


@pytest.mark.skip
def test_download_image(pinterest_downloader):
    with pytest.raises(ValueError):
        pinterest_downloader.download_image(image_url="dummy")

    pinterest_downloader.download_image(
        image_url="https://de.pinterest.com/pin/397513104614841320/"
    )
    file_path = Path(pinterest_downloader.output_dir, "397513104614841320")
    assert file_path.exists()

    pinterest_downloader.download_image(
        image_url="https://i.pinimg.com/1200x/88/3c/ab/883cab882a4b67bc2aa05555c5ece333.jpg"
    )
    file_path = Path(
        pinterest_downloader.output_dir, "883cab882a4b67bc2aa05555c5ece333.jpg"
    )
    assert file_path.exists()
    with Image.open(file_path) as img:
        assert img.format == "JPEG"


@pytest.mark.skip
def test_load_board(pinterest_downloader):
    pinterest_downloader.setup_driver()
    pinterest_downloader.load_board()

    current_url = pinterest_downloader.driver.current_url
    assert "pinterest.com/julianhatwell/discrae/by-me" in current_url

    # Assert that image elements are present, indicating the board loaded
    images = pinterest_downloader.driver.find_elements(By.TAG_NAME, "img")
    assert len(images) > 0


@pytest.mark.skip
def test_get_data_grid_items(pinterest_downloader):
    pinterest_downloader.setup_driver()
    pinterest_downloader.load_board()
    dgis = pinterest_downloader.get_visible_pin_elements()
    assert type(dgis[0]) == WebElement
