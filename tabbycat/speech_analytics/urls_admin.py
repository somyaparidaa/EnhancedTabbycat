from django.urls import path

from . import views

urlpatterns = [
    path('',
        views.SpeechAnalyticsDashboardView.as_view(),
        name='speech-analytics-dashboard'),
    path('recordings/upload/',
        views.SpeechRecordingUploadView.as_view(),
        name='speech-recording-upload'),
    path('recordings/<int:recording_id>/status/',
        views.SpeechRecordingStatusCheckView.as_view(),
        name='speech-recording-status'),
]
