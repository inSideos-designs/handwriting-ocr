import dagster

from orchestration.assets.data_prep import cleaned_dataset
from orchestration.assets.training import trained_model

defs = dagster.Definitions(
    assets=[cleaned_dataset, trained_model],
)
