"""
Convenience singleton so the rest of the app can simply do:

    from app.settings import settings
"""

from app.config import get_settings

settings = get_settings()
