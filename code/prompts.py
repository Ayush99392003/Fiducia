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
- The minimum image evidence requirement for this type of claim.

Your task is to evaluate whether the submitted images support the claim.

Rules:
1. The images are the PRIMARY source of truth.
2. The chat transcript defines what needs to be verified.
3. Select only from the allowed values for each field.
4. Contextualize severity using the chat transcript. BE EXTREMELY CONSERVATIVE. It is acceptable to under-predict severity, but you must NEVER over-predict. Default to a lower severity unless the image shows absolutely irrefutable catastrophic damage.
5. Output ONLY the required JSON object — no extra commentary.

If multiple risk flags apply, choose the most critical one. History-based
flags will be appended automatically by the post-processing layer.
"""


def build_direct_user_message(
    claim_object: str,
    user_claim: str,
    evidence_requirement: str,
    image_ids: list[str],
) -> str:
    """Build the user-turn text for Strategy A (Direct).

    Args:
        claim_object: One of 'car', 'laptop', or 'package'.
        user_claim: Raw chat transcript text.
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
- The minimum image evidence requirement for this type of claim.

Your task is to evaluate whether the submitted images support the claim.

IMPORTANT: You MUST complete the reasoning_scratchpad field FIRST before
filling any other field. Use this exact order in your scratchpad:
  STEP 1 — VISUAL ANALYSIS: Describe what is visible in each image.
  STEP 2 — CLAIM EXTRACTION: State exactly what the customer is claiming.
  STEP 3 — EVIDENCE CHECK: Does the image meet the minimum evidence
            requirement? State the requirement and assess it.
  STEP 4 — DECISION: State your claim_status and explain why.

Rules:
1. The images are the PRIMARY source of truth.
2. The chat transcript defines what needs to be verified.
3. Select only from the allowed values for each field.
4. Contextualize severity using the chat transcript. BE EXTREMELY CONSERVATIVE. It is acceptable to under-predict severity, but you must NEVER over-predict. Default to a lower severity unless the image shows absolutely irrefutable catastrophic damage.
5. Output ONLY the required JSON object — no extra commentary.

If multiple risk flags apply, choose the most critical one. History-based
flags will be appended automatically by the post-processing layer.
"""


def build_cot_user_message(
    claim_object: str,
    user_claim: str,
    evidence_requirement: str,
    image_ids: list[str],
) -> str:
    """Build the user-turn text for Strategy B (Chain-of-Thought).

    Args:
        claim_object: One of 'car', 'laptop', or 'package'.
        user_claim: Raw chat transcript text.
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
        f"MINIMUM EVIDENCE REQUIREMENT:\n{evidence_requirement}\n\n"
        f"SUBMITTED IMAGES (in order): {ids_str}\n\n"
        "Complete your reasoning_scratchpad first (Steps 1-4), then "
        "return the required JSON output."
    )
