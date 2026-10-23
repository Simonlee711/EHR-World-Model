"""EHR World Model — data pipeline, world models, training, inference, rollout and counterfactuals."""
from .config import FeatureConfig, TrainConfig
from .features import FeatureManifest, build_feature_manifest
from .data import (
    load_meds_split, load_meds_cohort,
    build_enriched_transitions, build_enriched_sequence_batches, EnrichedSequenceDataset,
)
from .labels import DISEASE_PREFIXES, extract_patient_labels, extract_disease_flags, attach_labels_to_transitions
from .cohort import (
    load_meds_fast, compute_demographics, extract_outcome_labels, extract_ehrshot_labels,
    build_label_table, LabeledSequenceDataset, DemoAugLoader, build_cohort, default_feature_config,
)
from .models import VelocityWorldModel, TransformerWorldModel, TRANSFORMER_SIZES
from .training import train_curriculum, CURRICULUM_STAGES, CURRICULUM_STAGES_FAST
from .inference import (
    get_device, build_model, save_checkpoint, load_checkpoint,
    build_patient_trajectories, train_risk_probe, get_traj_risk_series, get_traj_risk_scores,
)
from .rollout import evaluate_multistep_rollout, patient_timelines
from .counterfactual import (
    collect_patient_cf_data, free_rollout, counterfactual_rollouts, screen_single_drugs,
    greedy_mpc_rollout, shooting_mpc_rollout, observed_rollout,
)
