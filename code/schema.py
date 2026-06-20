"""Data models (Pydantic schemas) for structured VLM output.

Strategy A — DirectDamageClaimEvaluation: no reasoning scratchpad.
Strategy B — DamageClaimEvaluation (CoT): scratchpad forced as first field
    so the model reasons step-by-step before committing to categorical values.
"""

from typing import Literal, Type

from pydantic import BaseModel, Field, create_model

# ---------------------------------------------------------------------------
# Shared type aliases
# ---------------------------------------------------------------------------

ClaimStatus = Literal["supported", "contradicted", "not_enough_information"]

IssueType = Literal[
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
]

CarPart = Literal[
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "windshield",
    "side_mirror",
    "headlight",
    "taillight",
    "fender",
    "quarter_panel",
    "body",
    "unknown",
]

LaptopPart = Literal[
    "screen",
    "keyboard",
    "trackpad",
    "hinge",
    "lid",
    "corner",
    "port",
    "base",
    "body",
    "unknown",
]

PackagePart = Literal[
    "box",
    "package_corner",
    "package_side",
    "seal",
    "label",
    "contents",
    "item",
    "body",
    "unknown",
]

ObjectPartMap = {
    "car": CarPart,
    "laptop": LaptopPart,
    "package": PackagePart,
}

RiskFlag = Literal[
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
]

Severity = Literal["none", "low", "medium", "high", "unknown"]

# ---------------------------------------------------------------------------
# Shared field definitions (used by both strategies)
# ---------------------------------------------------------------------------

_EVIDENCE_MET = Field(
    description=(
        "True if the image set is sufficient to evaluate the claim "
        "based on the evidence requirements; otherwise False."
    )
)
_EVIDENCE_MET_REASON = Field(
    description="Short reason for the evidence decision."
)
_VALID_IMAGE = Field(
    description=(
        "True if the image set is usable for automated review; "
        "otherwise False."
    )
)
_RISK_FLAGS = Field(
    description=(
        "Primary risk flag identified in the image or user history. "
        "Use 'none' if no risk is present. The post-processing layer "
        "will append additional history-based flags automatically."
    )
)
_ISSUE_TYPE = Field(description="The visible issue type found in the image.")

def _get_object_part_field(claim_object: str):
    """Return a typed field for object_part based on the claim_object."""
    part_type = ObjectPartMap.get(claim_object.lower(), Literal["unknown"])
    return (part_type, Field(description="The specific part of the object being evaluated."))
_CLAIM_STATUS = Field(
    description=(
        "Final decision on whether the image evidence supports the "
        "user's claim."
    )
)
_CLAIM_STATUS_JUSTIFICATION = Field(
    description=(
        "Concise image-grounded explanation for the claim status "
        "decision. Mention relevant image IDs where helpful."
    )
)
_SUPPORTING_IMAGE_IDS = Field(
    description=(
        "Image IDs supporting the decision, separated by semicolons. "
        "GUARDRAIL: If claim is SUPPORTED, list ONLY the image(s) where the damage is clearly visible. Do NOT include context-only wide shots. "
        "If CONTRADICTED (e.g., intact package), list all images that prove it is intact. "
        "Use 'none' if no image is sufficient."
    )
)
_SEVERITY = Field(
    description=(
        "The estimated severity of the visible damage. STRICT GUIDELINES (BE EXTREMELY CONSERVATIVE - NEVER OVER-PREDICT):\n"
        "- none: No physical damage is visible in the image.\n"
        "- low: Minor cosmetic damage (e.g., surface scratches, small corner dents, minor box creases).\n"
        "- medium: Clearly visible damage affecting integrity but not catastrophic (e.g., standard dents, cracked glass, broken hinges, liquid stains, crushed packaging).\n"
        "- high: Severe, catastrophic structural damage or complete detachment.\n"
        "- unknown: Image does not show the required part or lacks information."
    )
)


# ---------------------------------------------------------------------------
# Strategy A — Direct Prediction (Normal)
# ---------------------------------------------------------------------------


def get_direct_schema(claim_object: str) -> Type[BaseModel]:
    """Dynamically generate the Strategy A schema locked to a specific claim object."""
    part_type, part_field = _get_object_part_field(claim_object)
    
    return create_model(
        "DirectDamageClaimEvaluation",
        evidence_standard_met=(bool, _EVIDENCE_MET),
        evidence_standard_met_reason=(str, _EVIDENCE_MET_REASON),
        valid_image=(bool, _VALID_IMAGE),
        risk_flags=(RiskFlag, _RISK_FLAGS),
        issue_type=(IssueType, _ISSUE_TYPE),
        object_part=(part_type, part_field),
        claim_status=(ClaimStatus, _CLAIM_STATUS),
        claim_status_justification=(str, _CLAIM_STATUS_JUSTIFICATION),
        supporting_image_ids=(str, _SUPPORTING_IMAGE_IDS),
        severity=(Severity, _SEVERITY),
    )


# ---------------------------------------------------------------------------
# Strategy B — Chain-of-Thought (CoT)
# ---------------------------------------------------------------------------


def get_cot_schema(claim_object: str) -> Type[BaseModel]:
    """Dynamically generate the Strategy B schema locked to a specific claim object."""
    part_type, part_field = _get_object_part_field(claim_object)
    
    _REASONING = Field(
        description=(
            "Step-by-step reasoning block (not included in output.csv). "
            "1. Describe exactly what is visible in each image. "
            "2. Extract the user's claim from the chat transcript. "
            "3. Cross-reference the minimum evidence requirement for "
            "this claim_object and damage type. "
            "4. Determine the claim_status and all output fields."
        )
    )

    return create_model(
        "DamageClaimEvaluation",
        reasoning_scratchpad=(str, _REASONING),
        evidence_standard_met=(bool, _EVIDENCE_MET),
        evidence_standard_met_reason=(str, _EVIDENCE_MET_REASON),
        valid_image=(bool, _VALID_IMAGE),
        risk_flags=(RiskFlag, _RISK_FLAGS),
        issue_type=(IssueType, _ISSUE_TYPE),
        object_part=(part_type, part_field),
        claim_status=(ClaimStatus, _CLAIM_STATUS),
        claim_status_justification=(str, _CLAIM_STATUS_JUSTIFICATION),
        supporting_image_ids=(str, _SUPPORTING_IMAGE_IDS),
        severity=(Severity, _SEVERITY),
    )
