"""Frozen-model coordinate enrichment from an independently matched OSM snapshot.

This module never learns from prices or refits the archived transformer.  The
registry adds only missing houses; fitted coordinates and trait guards survive.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from building_catalog import ASTANA_BBOX
from location_model import LOCATION_INPUT_COLUMNS, prepare_location_input


POLICY_VERSION = "osm_exact_address_missing_houses_v1"
CANDIDATE_ID = "complex_coords_osm_exact_address_v1"
_OSM_URL = re.compile(r"https://(?:www\.)?openstreetmap\.org/(?:node|way|relation)/[1-9][0-9]*/?\Z")
_TRAITS = ("residential_complex", "year_built", "total_floors")


@dataclass(frozen=True)
class VerifiedRegistry:
    """Validated local snapshot; source SHA covers all provenance, not just points."""

    entries: dict[str, dict[str, Any]]
    sha256: str
    path: str
    document: dict[str, Any]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Повторный JSON-ключ в реестре: {key}.")
        result[key] = value
    return result


def _validate_entries(document: dict) -> dict[str, dict[str, Any]]:
    if (not isinstance(document, dict) or document.get("schema_version") != 1
            or not isinstance(document.get("entries"), dict)):
        raise ValueError("Реестр должен иметь schema_version=1 и объект entries.")
    entries = copy.deepcopy(document["entries"])
    for key, entry in entries.items():
        if (not isinstance(key, str) or key != key.strip() or len(key.split("|")) != 3
                or key.split("|")[0] != "astana" or not all(key.split("|"))):
            raise ValueError("Реестр содержит некорректный канонический house_key.")
        if (not isinstance(entry, dict) or entry.get("source_status") != "osm_exact_address"
                or not isinstance(entry.get("osm_url"), str) or not _OSM_URL.fullmatch(entry["osm_url"])):
            raise ValueError(f"Дом {key}: нужен точный адрес OSM и безопасная ссылка на объект.")
        if "eligible_for_enrichment" in entry and entry["eligible_for_enrichment"] is not True:
            raise ValueError(f"Дом {key}: запись не допущена к обогащению модели.")
        for column in ("latitude", "longitude"):
            value = entry.get(column)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Дом {key}: координаты должны быть конечными числами.")
            if not ASTANA_BBOX[column + "_min"] <= value <= ASTANA_BBOX[column + "_max"]:
                raise ValueError(f"Дом {key}: координаты за пределами Астаны.")
            entry[column] = float(value)
    return entries


def load_verified_registry(path: Path) -> VerifiedRegistry:
    """Read once and validate; network geocoding is deliberately outside inference."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object)
    return VerifiedRegistry(_validate_entries(document), hashlib.sha256(raw).hexdigest(), str(path), document)


def _trait(value: Any, column: str) -> str | float | None:
    if column == "residential_complex":
        return value.strip() or None if isinstance(value, str) else None
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


class VerifiedGeoOverlay(TransformerMixin, BaseEstimator):
    """An inference-only wrapper, retaining the archived 13-column input contract."""

    def __init__(self, base_transformer: BaseEstimator, registry: VerifiedRegistry):
        self.base_transformer = base_transformer
        self.registry = registry

    @property
    def variant(self) -> str:
        return "complex_coords"

    def fit(self, X: Any, y: Any = None) -> "VerifiedGeoOverlay":
        raise RuntimeError("VerifiedGeoOverlay — замороженный вариант: fit запрещён.")

    def initialize(self) -> "VerifiedGeoOverlay":
        base = self.base_transformer
        if (getattr(base, "variant", None) != "complex_coords"
                or list(getattr(base, "feature_names_in_", [])) != list(LOCATION_INPUT_COLUMNS)
                or not hasattr(base, "house_traits_") or not hasattr(base, "coordinate_audit_")):
            raise ValueError("Нужен обученный исходный transformer complex_coords.")
        if isinstance(base, VerifiedGeoOverlay):
            raise ValueError("Повторное обогащение уже изменённой модели запрещено.")
        entries = _validate_entries(self.registry.document)
        if entries != self.registry.entries:
            raise ValueError("Реестр изменён после загрузки.")
        self.feature_names_in_ = np.asarray(LOCATION_INPUT_COLUMNS, dtype=object)
        self.n_features_in_ = len(LOCATION_INPUT_COLUMNS)
        self.house_traits_ = copy.deepcopy(base.house_traits_)
        self.blocked_house_keys_ = set(base.coordinate_audit_.loc[
            base.coordinate_audit_.coordinate_status.eq("trait_conflict"), "house_key"
        ])
        self.external_entries_ = {
            key: copy.deepcopy(entry) for key, entry in entries.items()
            if key not in base.house_coordinate_map_ and key not in self.blocked_house_keys_
        }
        self.house_coordinate_map_ = copy.deepcopy(base.house_coordinate_map_)
        self.house_coordinate_map_.update({key: (entry["latitude"], entry["longitude"])
                                           for key, entry in self.external_entries_.items()})
        self.registry_sha256_ = self.registry.sha256
        self.policy_version_ = POLICY_VERSION
        return self

    def provenance(self, house_key: str | None) -> dict[str, Any]:
        """House-level provenance; row-specific trait guards are checked separately."""
        check_is_fitted(self, "external_entries_")
        if house_key in self.base_transformer.house_coordinate_map_:
            return {"source_status": "training_consensus", "externally_matched": False}
        if house_key in self.blocked_house_keys_:
            return {"source_status": "blocked_trait_conflict", "externally_matched": False}
        if house_key in self.external_entries_:
            return {**copy.deepcopy(self.external_entries_[house_key]), "externally_matched": True}
        return {"source_status": "missing", "externally_matched": False}

    def _eligible(self, row: dict[str, Any]) -> bool:
        key = row["house_key"]
        if key not in self.external_entries_:
            return False
        known = self.house_traits_.get(key, {})
        return not any(known.get(column) is not None and _trait(row[column], column) is not None
                       and known[column] != _trait(row[column], column) for column in _TRAITS)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "external_entries_")
        frame = prepare_location_input(X)
        result = self.base_transformer.transform(frame).copy(deep=True)
        coordinates = result[["latitude", "longitude"]].to_numpy(dtype=float, copy=True)
        for position, row in enumerate(frame.to_dict("records")):
            if self._eligible(row):
                entry = self.external_entries_[row["house_key"]]
                coordinates[position] = entry["latitude"], entry["longitude"]
        result[["latitude", "longitude"]] = coordinates
        return result

    def coordinate_provenance(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = prepare_location_input(X)
        transformed = self.transform(frame)
        records = []
        for position, row in enumerate(frame.to_dict("records")):
            available = bool(pd.notna(transformed.latitude.iloc[position]))
            info = self.provenance(row["house_key"])
            records.append({"house_key": row["house_key"], "coordinate_available": available,
                            "source_status": info["source_status"] if available else
                            ("query_trait_conflict" if row["house_key"] in self.house_coordinate_map_ else info["source_status"]),
                            "externally_matched": bool(available and self._eligible(row)),
                            "osm_url": info.get("osm_url") if available else None})
        return pd.DataFrame(records, index=frame.index,
                            columns=["house_key", "coordinate_available", "source_status", "externally_matched", "osm_url"])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "external_entries_")
        return self.base_transformer.get_feature_names_out(input_features)


def build_verified_candidate(base_pipeline: Pipeline, registry: VerifiedRegistry) -> Pipeline:
    """Deep-copy fitted components; no shared mutable estimator and no fit call."""
    if not isinstance(base_pipeline, Pipeline) or set(base_pipeline.named_steps) != {"location_features", "preprocess", "model"}:
        raise ValueError("Нужен исходный location Pipeline.")
    candidate = copy.deepcopy(base_pipeline)
    overlay = VerifiedGeoOverlay(candidate.named_steps["location_features"], copy.deepcopy(registry)).initialize()
    candidate.steps[0] = ("location_features", overlay)
    return candidate
