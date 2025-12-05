import json
from playwright.async_api import Page # Import the Page type for type hinting

# --- 1. Load Configuration ---
# Note: Use an absolute or relative path that works from where your test script is run.
try:
    with open("config.json", "r") as f:
        # Load the JSON data into a Python dictionary
        config = json.load(f)
except FileNotFoundError:
    # Handle the case where the config file is not found
    print("Error: config.json not found.")
    config = {}

# --- 2. Page Object Class (using Async API) ---
class LoginPage:
    
    # Python constructor uses __init__
    def __init__(self, page: Page):
        # Store the Playwright Page object
        self.page = page
        
        # Optional: Define locators as attributes for cleaner code (Recommended POM practice)
        self.endpoint = config.get("endpoint")
        self.username_input = config.get("username_field")
        self.password_input = config.get("password_field")
        self.login_btn = config.get("login_button")

    # Python async method uses async def
    async def navigate(self):
        # Await the Playwright page.goto() method
        await self.page.goto(self.endpoint)
    
    # Python async method for login action
    async def login(self, username: str, password: str):
        # Await the page.fill() method using the stored locators
        await self.page.fill(self.username_input, username)
        await self.page.fill(self.password_input, password)
        # Await the page.click() method
        await self.page.click(self.login_btn)

    # Python async method to retrieve inner text
    async def get_inner_text(self):
        # Await the page.inner_text() method
        # Note: Playwright Python uses snake_case, e.g., inner_text
        return await self.page.inner_text("p")