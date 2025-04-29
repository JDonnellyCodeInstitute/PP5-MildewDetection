import streamlit as st

class MultiPage:
    def __init__(self, app_name: str) -> None:
        self.pages = []
        self.app_name = app_name
        st.set_page_config(page_title=self.app_name, page_icon="🍒")

    def add_page(self, title: str, func) -> None:
        """Adds a page to the sidebar."""
        self.pages.append({"title": title, "function": func})

    def run(self):
        """Renders the sidebar and runs the selected page."""
        st.title(self.app_name)
        page = st.sidebar.radio("Navigation", self.pages, format_func=lambda p: p["title"])
        page["function"]()