"""System and user prompt templates for the VLM damage-claim evaluator.

Strategy A — DIRECT_SYSTEM_PROMPT / build_direct_user_message:
    No intermediate reasoning; model outputs classifications directly.

Strategy B — COT_SYSTEM_PROMPT / build_cot_user_message:
    Model writes a step-by-step reasoning_scratchpad first, then outputs
    all classification fields.
"""

# ---------------------------------------------------------------------------
# Strategy A — Direct / Normal prompt
# ---------------------------------------------------------------------------

DIRECT_SYSTEM_PROMPT = """\
You are a damage claim verification specialist for an insurance platform.

You will receive:
- A customer chat transcript describing a damage claim.
- One or more submitted images (encoded as base64).
- The user's claim history risk summary.
- The minimum image evidence requirement for this type of claim.

Your task is to evaluate whether the submitted images support the claim.

Rules:
1. The images are the PRIMARY source of truth.
2. The chat transcript defines what needs to be verified.
3. User history adds risk context but does NOT override clear visual evidence.
4. Select only from the allowed values for each field.
5. Output ONLY the required JSON object — no extra commentary.

Allowed values:
- claim_status: supported | contradicted | not_enough_information
- issue_type: dent | scratch | crack | glass_shatter | broken_part | \
missing_part | torn_packaging | crushed_packaging | water_damage | stain \
| none | unknown
- severity: none | low | medium | high | unknown
- risk_flags (single primary flag): none | blurry_image | \
cropped_or_obstructed | low_light_or_glare | wrong_angle | wrong_object | \
wrong_object_part | damage_not_visible | claim_mismatch | \
possible_manipulation | non_original_image | text_instruction_present | \
user_history_risk | manual_review_required

If multiple risk flags apply, choose the most critical one. History-based
flags will be appended automatically by the post-processing layer.
"""


def build_direct_user_message(
    claim_object: str,
    user_claim: str,
    history_summary: str,
    evidence_requirement: str,
    image_ids: list[str],
) -> str:
    """Build the user-turn text for Strategy A (Direct).

    Args:
        claim_object: One of 'car', 'laptop', or 'package'.
        user_claim: Raw chat transcript text.
        history_summary: Human-readable summary from user_history.csv.
        evidence_requirement: Minimum evidence text from
            evidence_requirements.csv.
        image_ids: Ordered list of image IDs (e.g. ['img_1', 'img_2']).

    Returns:
        Formatted user-turn string for the VLM.
    """
    ids_str = ", ".join(image_ids)
    return (
        f"CLAIM OBJECT: {claim_object}\n\n"
        f"CUSTOMER CHAT TRANSCRIPT:\n{user_claim}\n\n"
        f"USER HISTORY RISK SUMMARY:\n{history_summary}\n\n"
        f"MINIMUM EVIDENCE REQUIREMENT:\n{evidence_requirement}\n\n"
        f"SUBMITTED IMAGES (in order): {ids_str}\n\n"
        "Evaluate the claim and return the required JSON output."
    )


# ---------------------------------------------------------------------------
# Strategy B — Chain-of-Thought prompt
# ---------------------------------------------------------------------------

COT_SYSTEM_PROMPT = """\
You are a damage claim verification specialist for an insurance platform.

You will receive:
- A customer chat transcript describing a damage claim.
- One or more submitted images (encoded as base64).
- The user's claim history risk summary.
- The minimum image evidence requirement for this type of claim.

Your task is to evaluate whether the submitted images support the claim.

IMPORTANT: You MUST complete the reasoning_scratchpad field FIRST before
filling any other field. Use this exact order in your scratchpad:
  STEP 1 — VISUAL ANALYSIS: Describe what is visible in each image.
  STEP 2 — CLAIM EXTRACTION: State exactly what the customer is claiming.
  STEP 3 — EVIDENCE CHECK: Does the image meet the minimum evidence
            requirement? State the requirement and assess it.
  STEP 4 — HISTORY RISK: Note any user history risk signals.
  STEP 5 — DECISION: State your claim_status and explain why.

Rules:
1. The images are the PRIMARY source of truth.
2. The chat transcript defines what needs to be verified.
3. User history adds risk context but does NOT override clear visual evidence.
4. Select only from the allowed values for each field.
5. Output ONLY the required JSON object — no extra commentary.

Allowed values:
- claim_status: supported | contradicted | not_enough_information
- issue_type: dent | scratch | crack | glass_shatter | broken_part | \
missing_part | torn_packaging | crushed_packaging | water_damage | stain \
| none | unknown
- severity: none | low | medium | high | unknown
- risk_flags (single primary flag): none | blurry_image | \
cropped_or_obstructed | low_light_or_glare | wrong_angle | wrong_object | \
wrong_object_part | damage_not_visible | claim_mismatch | \
possible_manipulation | non_original_image | text_instruction_present | \
user_history_risk | manual_review_required

If multiple risk flags apply, choose the most critical one. History-based
flags will be appended automatically by the post-processing layer.
"""


def build_cot_user_message(
    claim_object: str,
    user_claim: str,
    history_summary: str,
    evidence_requirement: str,
    image_ids: list[str],
) -> str:
    """Build the user-turn text for Strategy B (Chain-of-Thought).

    Args:
        claim_object: One of 'car', 'laptop', or 'package'.
        user_claim: Raw chat transcript text.
        history_summary: Human-readable summary from user_history.csv.
        evidence_requirement: Minimum evidence text from
            evidence_requirements.csv.
        image_ids: Ordered list of image IDs (e.g. ['img_1', 'img_2']).

    Returns:
        Formatted user-turn string for the VLM.
    """
    ids_str = ", ".join(image_ids)
    return (
        f"CLAIM OBJECT: {claim_object}\n\n"
        f"CUSTOMER CHAT TRANSCRIPT:\n{user_claim}\n\n"
        f"USER HISTORY RISK SUMMARY:\n{history_summary}\n\n"
        f"MINIMUM EVIDENCE REQUIREMENT:\n{evidence_requirement}\n\n"
        f"SUBMITTED IMAGES (in order): {ids_str}\n\n"
        "Complete your reasoning_scratchpad first (Steps 1-5), then "
        "return the required JSON output."
    )
