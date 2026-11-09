from django.contrib import admin
from django.utils.translation import gettext_lazy as _

from utils.admin import ModelAdmin, TabbycatModelAdminFieldsMixin

from .models import SpeechRecording, SpeechTranscript


class SpeechTranscriptInline(admin.StackedInline):
    model = SpeechTranscript
    extra = 0
    readonly_fields = ('created_at', 'updated_at')
    classes = ('collapse',)


@admin.register(SpeechRecording)
class SpeechRecordingAdmin(TabbycatModelAdminFieldsMixin, ModelAdmin):
    list_display = (
        'id',
        'get_tournament',
        'get_round',
        'debate',
        'speaker',
        'status',
        'job_id',
        'created_at',
    )
    list_filter = ('status', 'debate__round__tournament')
    search_fields = (
        'job_id',
        'speaker__name',
        'debate__round__tournament__name',
        'error_message',
    )
    raw_id_fields = ('debate', 'speaker')
    readonly_fields = ('created_at', 'updated_at')
    inlines = [SpeechTranscriptInline]

    def get_queryset(self, request):
        return super().get_queryset(request).select_related(
            'debate__round__tournament',
            'speaker__team',
        )

    @admin.display(description=_("Tournament"))
    def get_tournament(self, obj):
        return obj.debate.round.tournament.short_name or obj.debate.round.tournament.name

    @admin.display(description=_("Round"))
    def get_round(self, obj):
        return obj.debate.round.abbreviation


@admin.register(SpeechTranscript)
class SpeechTranscriptAdmin(TabbycatModelAdminFieldsMixin, ModelAdmin):
    list_display = (
        'id',
        'recording',
        'language_code',
        'created_at',
    )
    list_filter = ('language_code',)
    search_fields = (
        'text',
        'recording__speaker__name',
    )
    raw_id_fields = ('recording',)
    readonly_fields = ('created_at', 'updated_at')
