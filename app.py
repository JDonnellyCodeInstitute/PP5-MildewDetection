import streamlit as st
from app_pages.multipage import MultiPage

# Load pages scripts
from app_pages.project_charter import page_project_charter_body

def main():
    # Create an instance of the app
    app = MultiPage(app_name="Cherry Leaf Mildew Detector")

    # Add app pages
    app.add_page("Project Charter", page_project_charter_body)

    # Run app
    app.run()


if __name__ == "__main__":
    main()