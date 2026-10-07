"""Append-only review events for human corrections."""

from __future__ import annotations

from datetime import UTC, datetime

from .contracts import ReviewEvent, ReviewEventType
from .ids import make_review_event_id


def make_review_event(
    *,
    segment_id: str,
    event_type: ReviewEventType,
    field_name: str,
    previous_value: str | None,
    new_value: str | None,
    reviewer_id: str | None = None,
    reason: str | None = None,
) -> ReviewEvent:
    return ReviewEvent(
        event_id=make_review_event_id(),
        segment_id=segment_id,
        event_type=event_type,
        field_name=field_name,
        previous_value=previous_value,
        new_value=new_value,
        reviewer_id=reviewer_id,
        reason=reason,
        created_at=datetime.now(UTC),
    )
