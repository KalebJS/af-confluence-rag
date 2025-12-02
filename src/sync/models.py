"""Data models for synchronization operations."""

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field

from src.models.page import Page


class ChangeSet(BaseModel):
    """Represents detected changes in a Confluence space."""

    new_pages: list[Page] = Field(
        default_factory=list, description="Pages that are new and need to be added"
    )
    modified_pages: list[Page] = Field(
        default_factory=list, description="Pages that have been modified and need updating"
    )
    deleted_page_ids: list[str] = Field(
        default_factory=list, description="Page IDs that have been deleted"
    )

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes to process."""
        return bool(self.new_pages or self.modified_pages or self.deleted_page_ids)

    @property
    def total_changes(self) -> int:
        """Get total number of changes."""
        return len(self.new_pages) + len(self.modified_pages) + len(self.deleted_page_ids)


class SyncReport(BaseModel):
    """Report of synchronization operation results."""

    space_key: str = Field(..., description="Confluence space key that was synced")
    pages_added: int = Field(default=0, ge=0, description="Number of new pages added")
    pages_updated: int = Field(default=0, ge=0, description="Number of pages updated")
    pages_deleted: int = Field(default=0, ge=0, description="Number of pages deleted")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Sync duration in seconds")
    start_time: datetime = Field(..., description="Sync start timestamp")
    end_time: datetime = Field(..., description="Sync end timestamp")
    errors: list[str] = Field(
        default_factory=list, description="List of errors encountered during sync"
    )

    @property
    def total_changes(self) -> int:
        """Get total number of changes processed."""
        return self.pages_added + self.pages_updated + self.pages_deleted

    @property
    def success(self) -> bool:
        """Check if sync completed without errors."""
        return len(self.errors) == 0
