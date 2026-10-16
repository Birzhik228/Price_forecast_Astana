"""Train-only geographic consensus plus the same fixed OSM registry at fit/predict.

This is a separate training component.  The archived location transformer and
inference-only overlay are reused without changing their source or artifacts.
The external registry is a fixed input snapshot, never rebuilt during this fit.
Its earlier eligibility audit used non-price traits from the complete catalogue,
including held-out rows; callers must disclose that retrospective limitation.
"""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from location_model import LocationFeatureTransformer, make_location_pipeline, prepare_location_input
from verified_geo_model import VerifiedGeoOverlay, VerifiedRegistry


class TrainableVerifiedGeoTransformer(TransformerMixin, BaseEstimator):
    """Clone-friendly wrapper whose learned maps and trait guards see only fit X.

    ``coordinate_audit_`` describes the base train-only raw-point checks.  It
    must not be interpreted as an audit of all external addresses.  Effective
    per-row geography is available through ``coordinate_provenance(X)`` and
    separate train/external counts through ``training_geo_summary_``.
    """

    def __init__(self, registry: VerifiedRegistry):
        self.registry = registry

    @property
    def variant(self) -> str:
        return "complex_coords"

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TrainableVerifiedGeoTransformer":
        if not isinstance(self.registry, VerifiedRegistry):
            raise ValueError("registry должен быть загруженным VerifiedRegistry.")
        frame = prepare_location_input(X)
        # A fresh base per fit prevents maps from a prior fold/refit carrying over.
        # Neither the target nor unrelated catalogue fields reach its fit method.
        base = LocationFeatureTransformer(variant="complex_coords").fit(frame)
        overlay = VerifiedGeoOverlay(base, copy.deepcopy(self.registry)).initialize()
        provenance = overlay.coordinate_provenance(frame)
        train_houses = set(base.house_traits_)
        external_houses = set(overlay.external_entries_)
        summary = {
            **copy.deepcopy(base.training_geo_summary_),
            "coordinate_audit_scope": "fit_rows_raw_points_only",
            "external_registry_scope": "fixed_snapshot_independent_of_this_fit",
            "registry_sha256": overlay.registry_sha256_,
            "policy_version": overlay.policy_version_,
            "registry_house_count": len(overlay.registry.entries),
            "base_coordinate_house_count": len(base.house_coordinate_map_),
            "external_coordinate_house_count": len(external_houses),
            "combined_coordinate_house_count": len(overlay.house_coordinate_map_),
            "external_houses_present_in_train": len(external_houses & train_houses),
            "external_houses_not_present_in_train": len(external_houses - train_houses),
            "registry_houses_blocked_by_train_traits": len(set(overlay.registry.entries) & overlay.blocked_house_keys_),
            "registry_houses_with_preserved_train_coordinates": len(set(overlay.registry.entries) & set(base.house_coordinate_map_)),
            "training_consensus_rows": int(provenance.source_status.eq("training_consensus").sum()),
            "external_coordinate_train_rows": int(provenance.externally_matched.sum()),
            "available_coordinate_train_rows": int(provenance.coordinate_available.sum()),
            "missing_coordinate_train_rows": int((~provenance.coordinate_available).sum()),
            "fit_rows_without_house_key": int(frame.house_key.isna().sum()),
            "external_address_registry_applied": True,
            "external_match_is_cadastral_verification": False,
        }
        self.base_transformer_ = base
        self.overlay_ = overlay
        self.feature_names_in_ = overlay.feature_names_in_.copy()
        self.n_features_in_ = overlay.n_features_in_
        self.house_traits_ = overlay.house_traits_
        self.house_coordinate_map_ = overlay.house_coordinate_map_
        self.coordinate_audit_ = base.coordinate_audit_.copy(deep=True)
        self.registry_sha256_ = overlay.registry_sha256_
        self.policy_version_ = overlay.policy_version_
        self.training_geo_summary_ = summary
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "overlay_")
        return self.overlay_.transform(X)

    def coordinate_provenance(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "overlay_")
        return self.overlay_.coordinate_provenance(X)

    def provenance(self, house_key: str | None) -> dict[str, Any]:
        check_is_fitted(self, "overlay_")
        return self.overlay_.provenance(house_key)

    def get_feature_names_out(self, input_features: Any = None):
        check_is_fitted(self, "overlay_")
        return self.overlay_.get_feature_names_out(input_features)


def make_trainable_verified_pipeline(base_regressor: BaseEstimator, registry: VerifiedRegistry) -> Pipeline:
    """Use the unchanged location preprocessing/regressor factory, replacing only geography.

    The inherited TransformerMixin.fit_transform calls this component's fit and
    then its overlay-aware transform, so the model sees external points already
    during fitting. ``make_location_pipeline`` retains its n_jobs=2 policy.
    """
    pipeline = make_location_pipeline(base_regressor, "complex_coords")
    pipeline.steps[0] = ("location_features", TrainableVerifiedGeoTransformer(registry))
    return pipeline
