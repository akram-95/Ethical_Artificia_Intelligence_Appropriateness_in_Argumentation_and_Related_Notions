"""
ASGI config for Ethical_Artificia_Intelligence_Appropriateness_in_Argumentation_and_Related_Notions project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Ethical_Artificia_Intelligence_Appropriateness_in_Argumentation_and_Related_Notions.settings')

application = get_asgi_application()
