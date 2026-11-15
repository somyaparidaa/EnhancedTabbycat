"""Validation and contract layer for AI-powered competitive debate analysis.
Provides schema validation, domain checks, and evidence integrity verification
independent of the underlying LLM provider.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import jsonschema
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError

SCHEMA_DIR = os.path.join(os.path.dirname(__file__), 'schemas')
V1_SCHEMA_PATH = os.path.join(SCHEMA_DIR, 'debate_analysis_v1.json')


class DebateAnalysisValidationError(Exception):
    """Raised when an analysis payload violates the schema or integrity rules."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.message = message
        self.errors = errors or [message]


def load_debate_analysis_schema(version: str = "1.0.0") -> Dict[str, Any]:
    """Load the JSON schema dictionary for the requested contract version."""
    if version in ("1.0.0", "1.0", "v1"):
        schema_path = V1_SCHEMA_PATH
    else:
        raise ValueError(f"Unsupported debate analysis contract version: {version}")

    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_debate_analysis(
    payload: Any,
    version: str = "1.0.0",
    check_evidence_integrity: bool = True,
) -> Tuple[bool, List[str]]:
    """Validate a debate analysis dictionary against the versioned JSON Schema

    Args:
        payload: The analysis dictionary to validate.
        version: The schema version string (defaults to 1.0.0).
        check_evidence_integrity: When True, checks logical constraints such as
            start_time <= end_time and non-empty quote strings across all evidence.

    Returns:
        A tuple of (is_valid: bool, error_messages: List[str]).
    """
    errors: List[str] = []

    if not isinstance(payload, dict):
        return False, [f"Payload must be a dictionary, got {type(payload).__name__}."]

    schema = load_debate_analysis_schema(version)

    # 1. Structural validation via JSON Schema (Draft 2020-12 validator)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator = validator_cls(schema)

    for err in sorted(validator.iter_errors(payload), key=lambda e: e.path):
        loc = ".".join(str(p) for p in err.path) if err.path else "root"
        errors.append(f"[{loc}] {err.message}")

    # If structural errors already exist, return them early
    if errors:
        return False, errors

    # 2. Domain & Evidence Integrity Validation
    if check_evidence_integrity:
        evidence_errors = _validate_evidence_integrity(payload)
        errors.extend(evidence_errors)

    return (len(errors) == 0), errors


def ensure_valid_debate_analysis(
    payload: Any,
    version: str = "1.0.0",
    check_evidence_integrity: bool = True,
) -> Dict[str, Any]:
    """Raise DebateAnalysisValidationError if the analysis payload is invalid,
    otherwise return the payload.
    """
    is_valid, errors = validate_debate_analysis(
        payload,
        version=version,
        check_evidence_integrity=check_evidence_integrity,
    )
    if not is_valid:
        raise DebateAnalysisValidationError(
            f"Debate analysis payload failed validation ({len(errors)} error(s)): {'; '.join(errors[:3])}",
            errors=errors,
        )
    return payload


def _validate_evidence_integrity(payload: Dict[str, Any]) -> List[str]:
    """Inspect all EvidenceReference objects embedded throughout the sections."""
    errors: List[str] = []

    def check_evidence_list(ev_list: Any, section_path: str):
        if not isinstance(ev_list, list):
            return
        for idx, ev in enumerate(ev_list):
            if not isinstance(ev, dict):
                continue
            item_path = f"{section_path}.evidence[{idx}]"

            # 1. Timestamp order sanity
            start_time = ev.get('start_time')
            end_time = ev.get('end_time')
            if start_time is not None and end_time is not None:
                if start_time > end_time:
                    errors.append(
                        f"[{item_path}] start_time ({start_time}) cannot be greater than end_time ({end_time})."
                    )

            # 2. Quote string must not be purely whitespace
            quote = ev.get('quote')
            if isinstance(quote, str) and not quote.strip():
                errors.append(f"[{item_path}] quote cannot be empty or whitespace only.")

    # Check evidence across sections
    for i, arg in enumerate(payload.get('argument_analysis', [])):
        if isinstance(arg, dict):
            check_evidence_list(arg.get('evidence'), f"argument_analysis[{i}]")

    for i, reb in enumerate(payload.get('rebuttal_analysis', [])):
        if isinstance(reb, dict):
            check_evidence_list(reb.get('evidence'), f"rebuttal_analysis[{i}]")

    for i, clash in enumerate(payload.get('clash_analysis', [])):
        if isinstance(clash, dict):
            check_evidence_list(clash.get('evidence'), f"clash_analysis[{i}]")

    role_f = payload.get('role_fulfilment')
    if isinstance(role_f, dict):
        check_evidence_list(role_f.get('evidence'), "role_fulfilment")

    struct = payload.get('structure_analysis')
    if isinstance(struct, dict):
        check_evidence_list(struct.get('evidence'), "structure_analysis")

    return errors
