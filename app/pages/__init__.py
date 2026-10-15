"""app/pages/__init__.py"""
from .identify import render_identify_page
from .batch_analysis import render_batch_page
from .biodiversity import render_biodiversity_page
from .about import render_about_page
from .evaluation import render_evaluation_page

__all__ = [
    "render_identify_page",
    "render_batch_page",
    "render_biodiversity_page",
    "render_about_page",
    "render_evaluation_page"
]
