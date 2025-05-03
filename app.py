import streamlit as st
from app_pages.multipage import MultiPage

# Load pages scripts
from app_pages.project_charter import page_project_charter_body
from app_pages.leaf_atlas import page_leaf_atlas_body
from app_pages.preprocessing_playground import page_preprocessing_playground_body
from app_pages.training_dashboard import page_training_dashboard_body

def main():
    # Create an instance of the app
    app = MultiPage(app_name="Cherry Leaf Mildew Detector")

    # Add app pages
    app.add_page("Project Charter", page_project_charter_body)
    app.add_page("Leaf Atlas", page_leaf_atlas_body)
    app.add_page("Preprocessing Playground", page_preprocessing_playground_body)
    app.add_page("Training Dashboard", page_training_dashboard_body)

    # Run app
    app.run()


if __name__ == "__main__":
    main()