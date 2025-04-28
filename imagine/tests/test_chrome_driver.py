from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def test_setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage") # Recommended for Docker/Linux
    chrome_options.add_argument("--headless") # If you don't need a visible browser
    driver = webdriver.Chrome(options=chrome_options)
    print("Driver started successfully!") # Add this for debugging
    driver.quit()
