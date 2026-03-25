import dagster

from orchestration.assets.data_prep import cleaned_dataset
from orchestration.assets.training import trained_model


trained_model_k8s = trained_model.with_attributes(
    op_tags={
        "dagster-k8s/config": {
            "container_config": {
                "resources": {
                    "requests": {"nvidia.com/gpu": "1", "memory": "8Gi", "cpu": "3"},
                    "limits": {"nvidia.com/gpu": "1", "memory": "12Gi", "cpu": "4"},
                },
            },
            "pod_spec_config": {
                "node_selector": {
                    "cloud.google.com/gke-accelerator": "nvidia-tesla-t4",
                },
                "tolerations": [
                    {
                        "key": "nvidia.com/gpu",
                        "operator": "Exists",
                        "effect": "NoSchedule",
                    }
                ],
            },
        }
    },
)

defs = dagster.Definitions(
    assets=[cleaned_dataset, trained_model_k8s],
)
