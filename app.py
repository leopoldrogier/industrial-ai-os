"""Industrial AI OS — Streamlit entry point.

    streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Industrial AI OS", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

from src.ui import components as ui  # noqa: E402
from src.ui.pages import copilot, dashboard, inventory, opportunities, purchasing, suppliers  # noqa: E402
from src.ui.state import get_result  # noqa: E402

pages = {
    "dashboard": st.Page(dashboard.render, title="Dashboard", icon=":material/space_dashboard:",
                         url_path="dashboard", default=True),
    "suppliers": st.Page(suppliers.render, title="Supplier Performance", icon=":material/factory:", url_path="suppliers"),
    "inventory": st.Page(inventory.render, title="Inventory Intelligence", icon=":material/inventory_2:",
                         url_path="inventory"),
    "purchasing": st.Page(purchasing.render, title="Purchasing Intelligence", icon=":material/shopping_cart:",
                          url_path="purchasing"),
    "opportunities": st.Page(opportunities.render, title="Cost & Opportunities", icon=":material/savings:",
                             url_path="opportunities"),
    "copilot": st.Page(copilot.render, title="AI Industrial Copilot", icon=":material/smart_toy:", url_path="copilot"),
}
ui.PAGES.update(pages)

nav = st.navigation({
    "Overview": [pages["dashboard"]],
    "Intelligence modules": [pages["suppliers"], pages["inventory"], pages["purchasing"], pages["opportunities"]],
    "AI layer": [pages["copilot"]],
})
ui.inject_css()
ui.sidebar(get_result())
nav.run()
