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
    path('recordings/<int:recording_id>/retry/',
        views.SpeechRecordingRetryView.as_view(),
        name='speech-recording-retry'),
    path('recordings/<int:recording_id>/transcript/',
        views.SpeechTranscriptDetailView.as_view(),
        name='speech-transcript-detail'),
    path('recordings/<int:recording_id>/audio/',
        views.SpeechRecordingAudioStreamView.as_view(),
        name='speech-recording-audio'),
]
