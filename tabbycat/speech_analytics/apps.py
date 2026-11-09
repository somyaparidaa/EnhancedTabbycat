from django.apps import AppConfig
from django.utils.translation import gettext_lazy as _


class SpeechAnalyticsConfig(AppConfig):
    name = 'speech_analytics'
    verbose_name = _("Speech Analytics")
