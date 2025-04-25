import os
import time
import requests
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urlparse, unquote


class PinterestBoardDownloader:
    def __init__(self, output_dir="pinterest_images"):
        """Initialize the Pinterest board image downloader"""
        self.output_dir = output_dir
        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Setup Selenium with Chrome
        self.setup_driver()

    def setup_driver(self):
        """Set up the Chrome webdriver with necessary options"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")

        # Initialize the Chrome driver
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

    def extract_board_name(self, url):
        """Extract board name from URL for naming the folder"""
        path_parts = urlparse(url).path.strip("/").split("/")
        if len(path_parts) >= 2:
            return f"{path_parts[0]}_{path_parts[1]}"
        return "pinterest_board"

    def download_image(self, img_url, filename):
        """Download an image from a URL and save it to the specified filename"""
        try:
            response = requests.get(img_url, stream=True)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"Downloaded: {filename}")
                return True
            else:
                print(
                    f"Failed to download {img_url}: Status code {response.status_code}"
                )
                return False
        except Exception as e:
            print(f"Error downloading {img_url}: {e}")
            return False

    def extract_image_urls(self):
        """Extract image URLs from Pinterest board page"""
        # Wait for the image elements to load
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "img"))
            )
        except TimeoutException:
            print("Timeout waiting for images to load")

        # Find all image elements
        images = self.driver.find_elements(By.TAG_NAME, "img")
        image_urls = []

        for img in images:
            try:
                src = img.get_attribute("src")
                if src and ("orig" in src or "originals" in src):
                    # This is likely a full-size Pinterest image
                    image_urls.append(src)
            except Exception as e:
                print(f"Error extracting image URL: {e}")

        return image_urls

    def scroll_to_load_all_images(self, max_scrolls=100):
        """Scroll down the page to load more images"""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        scroll_count = 0

        while scroll_count < max_scrolls:
            # Scroll down to the bottom
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )

            # Wait for new images to load
            time.sleep(2)

            # Calculate new scroll height and compare with last scroll height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                # If heights are the same, we've reached the bottom or no more content is loading
                break

            last_height = new_height
            scroll_count += 1
            print(f"Scrolled down {scroll_count} times to load more images...")

    def download_board(
        self, board_url, login_required=False, username=None, password=None
    ):
        """Main method to download all images from a Pinterest board"""
        # Create a specific folder for this board
        board_name = self.extract_board_name(board_url)
        board_dir = os.path.join(self.output_dir, board_name)
        if not os.path.exists(board_dir):
            os.makedirs(board_dir)

        # Navigate to the Pinterest board URL
        print(f"Accessing Pinterest board: {board_url}")
        self.driver.get(board_url)

        # Handle login if required (may need to be customized)
        if login_required and username and password:
            self.login_to_pinterest(username, password)

        # Scroll to load all images
        print("Scrolling to load images...")
        self.scroll_to_load_all_images(
            max_scrolls=15
        )  # Adjust max_scrolls based on board size

        # Extract image URLs
        print("Extracting image URLs...")
        image_urls = self.extract_image_urls()
        print(f"Found {len(image_urls)} images to download")

        # Download each image
        downloaded_count = 0
        for i, img_url in enumerate(image_urls):
            # Extract a name for the image from the URL or use a counter
            filename = f"pin_{i + 1}.jpg"

            # Try to get a better filename from the URL
            try:
                parsed_url = urlparse(img_url)
                path = unquote(parsed_url.path)
                if "/" in path:
                    path_parts = path.split("/")
                    for part in path_parts:
                        if (
                            ".jpg" in part.lower()
                            or ".jpeg" in part.lower()
                            or ".png" in part.lower()
                        ):
                            filename = part
                            break
            except:
                pass  # If extraction fails, use the default name

            # Ensure the filename is valid
            filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
            filepath = os.path.join(board_dir, filename)

            # Download the image
            if self.download_image(img_url, filepath):
                downloaded_count += 1

        print(
            f"\nDownload complete! Downloaded {downloaded_count} out of {len(image_urls)} images."
        )
        print(f"Images saved in: {os.path.abspath(board_dir)}")

        # Clean up
        self.driver.quit()
        return downloaded_count

    def login_to_pinterest(self, username, password):
        """Log in to Pinterest (if needed)"""
        try:
            # Wait for login button and click it
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "button[data-test-id='simple-login-button']")
                )
            ).click()

            # Enter email/username
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "email"))
            ).send_keys(username)

            # Enter password
            self.driver.find_element(By.ID, "password").send_keys(password)

            # Click login button
            self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

            # Wait for login to complete
            time.sleep(5)
            print("Logged in to Pinterest")
        except Exception as e:
            print(f"Error during login: {e}")
