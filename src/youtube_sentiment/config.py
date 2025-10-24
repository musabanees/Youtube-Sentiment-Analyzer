from typing import Any

import yaml
from pydantic import BaseModel

class Tags(BaseModel):
    """Model for MLflow tags."""

    git_sha: str # Git commit SHA for the current code version that helps in tracking experiments and reproducibility.
    branch: str
    run_id: str | None = None
    experiment_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Convert the Tags instance to a dictionary."""
        tags_dict = {}
        tags_dict["git_sha"] = self.git_sha
        tags_dict["branch"] = self.branch
        if self.run_id is not None:
            tags_dict["run_id"] = self.run_id
        if self.experiment_id is not None:
            tags_dict["experiment_id"] = self.experiment_id
        return tags_dict
