"""
WSGI config for Ethical_Artificia_Intelligence_Appropriateness_in_Argumentation_and_Related_Notions project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Ethical_Artificia_Intelligence_Appropriateness_in_Argumentation_and_Related_Notions.settings')

application = get_wsgi_application()
