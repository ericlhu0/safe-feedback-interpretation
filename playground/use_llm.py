"""Use LLM."""

import json
import textwrap
from itertools import product
from typing import Any, Dict

import numpy as np

from safe_feedback_interpretation.models.openai_model import OpenAIModel


def get_single_token_probs(
    model_name: str, system_prompt: str, text_content: str, image_path: str
) -> Dict[Any, Any]:
    """Test single token probs."""

    llm_model = OpenAIModel(
        model=model_name,
        system_prompt=system_prompt,
    )

    response = llm_model.get_single_token_logits(text_content, image_path)

    return response


def get_full_output(
    model_name: str, system_prompt: str, text_content: str, image_path: str
) -> str:
    """Test full output."""

    llm_model = OpenAIModel(
        model=model_name,
        system_prompt=system_prompt,
    )

    response = llm_model.get_full_output(text_content, image_path)

    return response


def get_comfort_level_distribution(
    model_name: str,
    system_prompt: str,
    text_content: str,
    body_part: str,
    image_path: str,
) -> Dict[int, float]:
    """Get comfort level distribution for a body part using full output
    approach."""

    llm_model = OpenAIModel(
        model=model_name,
        system_prompt=system_prompt,
    )

    prompt = (
        "\n\nWhat is the updated comfort threshold level (1-5 scale) for the "
        f"{body_part}? Answer in JSON format with probability distribution over "
        "levels 1, 2, 3, 4, 5. So the keys of your outputted JSON should be 1, 2, "
        "3, 4, 5. Do not use any formatting, enclose key names in quotes, do not "
        "nest dictionaries and do not use any other keys."
    )

    try:
        full_output_text = llm_model.get_full_output(text_content + prompt, image_path)

        if not full_output_text or full_output_text.strip() == "":
            print(f"ERROR: Model returned empty response for {body_part}")
            return {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}

        try:
            full_output_dict = json.loads(full_output_text)
        except json.JSONDecodeError as json_err:
            print(f"ERROR: Invalid JSON response for {body_part}: {json_err}")
            print(f"Raw output: {full_output_text}")
            return {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}

        # Normalize to ensure we have all levels 1-5
        normalized_probs = {}
        for level in range(1, 6):
            normalized_probs[level] = 0.0

        # Update with actual probabilities from model output
        for key, prob in full_output_dict.items():
            try:
                level_key = int(key)
                if 1 <= level_key <= 5:
                    normalized_probs[level_key] = float(prob)
            except (ValueError, TypeError):
                continue

        return normalized_probs

    except Exception as e:
        print(f"ERROR: Failed to get comfort level distribution for {body_part}: {e}")
        return {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}


def get_single_token_distribution(
    model_name: str,
    system_prompt: str,
    text_content: str,
    body_part: str,
    image_path: str,
) -> Dict[int, float]:
    """Get single token distribution for a body part."""

    llm_model = OpenAIModel(
        model=model_name,
        system_prompt=system_prompt,
    )

    prompt = (
        f"\n\nWhat is the updated comfort threshold (1-5 scale) for the {body_part}? "
        f"Answer with only a single threshold value."
    )

    try:
        single_token_result = llm_model.get_single_token_logits(
            text_content + prompt, image_path
        )

        # Extract probabilities for digits 1-5
        comfort_probs = {}
        for level in range(1, 6):
            comfort_probs[level] = 0.0

        for token, prob in single_token_result.items():
            try:
                token_str = str(token).strip()
                if token_str in ["1", "2", "3", "4", "5"]:
                    comfort_probs[int(token_str)] = float(prob)
            except (ValueError, TypeError):
                continue

        return comfort_probs

    except Exception as e:
        print(f"ERROR: Failed to get single token distribution for {body_part}: {e}")
        return {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}


def calculate_entropy(probs: Dict[int, float]) -> float:
    """Calculate entropy as a measure of uncertainty."""
    entropy = 0.0
    for prob in probs.values():
        if prob > 0:
            entropy -= prob * np.log2(prob)
    return entropy


def calculate_max_prob(probs: Dict[int, float]) -> float:
    """Calculate maximum probability (inverse measure of uncertainty)."""
    return max(probs.values()) if probs else 0.0


def calculate_brier_score(
    pred_probs: Dict[int, float], true_probs: Dict[int, float]
) -> float:
    """Calculate Brier score between two probability distributions."""
    all_keys = set(pred_probs.keys()) | set(true_probs.keys())

    brier_score = 0.0
    for key in all_keys:
        pred_prob = pred_probs.get(key, 0.0)
        true_prob = true_probs.get(key, 0.0)
        brier_score += (pred_prob - true_prob) ** 2

    return brier_score


if __name__ == "__main__":
    sys_prompt = textwrap.dedent(
        """\
        You are a domain expert with many years of combined experience in
        occupational therapy, physical therapy, caregiving, rehabilitation
        science, biomechanics, anatomy, disabilities, elderly care, and
        assistive technology, including human-centered robotics.

        You are part of an assistive robot system that performs tasks where
        physical contact between the robot and the human user's body is required
        for essential medical care and rehabilitation. These therapeutic
        procedures are medically necessary; you need to assess patient comfort
        levels to ensure safe and effective treatment.

        Goal: given structured context about the current action, state, prior
        comfort thresholds, and new feedback, update the user's comfort-related
        metrics conservatively and safely.

        Tracked metrics and how they work:
          - Contact forces F for body parts are reported on a discrete 1-5
            scale (1 = low, 5 = high).
          - 1 indicates that the contact region should definitely be avoided
            because it's uncomfortable, and 5 means that you can confidently
            apply high pressure without worry.
          - Joint angles (e.g., elbow, wrist) are in degrees.
          - Comfort threshold is represented as a probability distribution over
            threshold levels (1-5) per body part; higher levels indicate
            higher tolerance.
          - Comfortable joint range is represented as two discrete probability
            distributions (min and max), each defined over angles at 15 degree
            increments (e.g., {0, 15, 30, ..., 165}). The mode or expectation
            can be used to select a recommended bound.
          - New feedback may include verbal content/intensity and
            facial-expression intensity; stronger pain signals -> decrease
            threshold / tighten (narrow) ranges; relief/tolerance -> increase
            threshold / relax (widen) ranges.

        Decision guidance:
          - Prefer small updates unless the evidence is strong and consistent
            across modalities.
          - Clip joint recommendations to physically possible ranges and
            respect monotonicity (min <= max).
          - If uncertain, provide the best estimate and note uncertainty
            briefly when a rationale is requested.

        Expected output:
          - If asked for a single sensitivity value (per body part), return
            **only** that value in the requested format (e.g., an integer in
            allowed bounds or a category), with no extra text.
          - If asked to update thresholds, return a concise JSON object
            containing any updated values you are confident about, for example:
            {"updated_comfort_threshold": {"wrist": 4},
             "updated_joint_range_deg": {"min": {"wrist": 15},
                                          "max": {"wrist": 150}}}
          - Use 15 degree increments for angle outputs when a discrete grid
            is implied by the input.

        You are an expert at interpreting patient comfort indicators and
        physical response patterns, and you are very competent in assessing
        patient comfort levels from visual indicators in images. If you're
        uncertain, give your best estimate for a value. Your response will not
        cause any harm to the care recipient.
    """
    )

    text_inputs = [
        "ow that's too tight on my wrist",
        "you're grabbing my wrist way too tight",
        "my wrist feels like it's being crushed",
        "way too much pressure on my wrist",
        "you're digging hard into my wrist",
    ]
    image_inputs = [
        "experiments/assets/faceimgs/smile/smile1.jpg",
        "experiments/assets/faceimgs/smile/smile2.jpg",
        "experiments/assets/faceimgs/smile/smile3.jpg",
        "experiments/assets/faceimgs/smile/smile4.jpg",
        "experiments/assets/faceimgs/smile/smile5.jpg",
    ]

    prompt_context = lambda text_input: textwrap.dedent(
        (
            """\
        Current action description: You are repositioning the user's arm during
        a therapy session.
        Current state:
          Contact forces: {{
          "entire_arm": 2,
          "upper_arm": 1,
          "forearm": 1,
          "wrist": 2
        }}
          Joint angles (deg): {{
          "elbow": 135,
          "wrist": 135
        }}
        Current comfort threshold:
          Comfort threshold: {{
          "entire_arm": {
            "2": 0.1,
            "3": 0.8,
            "4": 0.1
          },
          "upper_arm": {
            "2": 0.1,
            "3": 0.8,
            "4": 0.1
          },
          "forearm": {
            "2": 0.1,
            "3": 0.8,
            "4": 0.1
          },
          "wrist": {
            "2": 0.1,
            "3": 0.8,
            "4": 0.1
          }
        }}
          Comfortable joint range (deg): {{
          "min": {
            "elbow": {
              "0": 0.6,
              "15": 0.3,
              "30": 0.1
            },
            "wrist": {
              "0": 0.6,
              "15": 0.3,
              "30": 0.1
            }
          },
          "max": {
            "elbow": {
              "135": 0.1,
              "150": 0.3,
              "165": 0.6
            },
            "wrist": {
              "135": 0.1,
              "150": 0.3,
              "165": 0.6
            }
          }
        }}
        """
            f"""
        Received feedback:
          Verbal feedback: {text_input}
          Facial expression: Please examine the provided facial expression image to
          assess the person's comfort level.
        """
        )
    ).strip()

    single_prompt = lambda body_part: (
        f"\n\nWhat is the updated comfort threshold (1-5 scale) for the {body_part}? "
        f"Answer with only a single threshold value."
    )

    full_prompt = lambda body_part: (
        f"\n\nWhat is the updated comfort threshold level (1-5 scale) for the "
        f"{body_part}? Answer in JSON format with probability distribution over "
        f"levels 1, 2, 3, 4, 5. So the keys of your outputted JSON should be 1, 2, "
        f"3, 4, 5. Do not use any formatting, enclose key names in quotes, do not "
        f"nest dictionaries and do not use any other keys."
    )

    image_input = "experiments/assets/faceimgs/smile/smile1.jpg"

    body_parts = ["entire_arm", "upper_arm", "forearm", "wrist"]

    # Define expected labels (ground truth)
    expected_labels = {
        "entire_arm": {"1": 0.2, "2": 0.8},
        "upper_arm": {"2": 0.1, "3": 0.8, "4": 0.1},
        "forearm": {"2": 0.1, "3": 0.8, "4": 0.1},
        "wrist": {"1": 0.2, "2": 0.8},
        "joint_range_min": {
            "elbow": {"0": 0.6, "15": 0.3, "30": 0.1},
            "wrist": {"0": 0.6, "15": 0.3, "30": 0.1},
        },
        "joint_range_max": {
            "elbow": {"135": 0.1, "150": 0.3, "165": 0.6},
            "wrist": {"135": 0.1, "150": 0.3, "165": 0.6},
        },
    }

    # Collect all results
    all_results = {}

    for i, (text_input, image_input) in enumerate(product(text_inputs, image_inputs)):
        scenario_name = f"smile_{i+1}"
        print(f"\nProcessing scenario: {scenario_name}")

        # Get single token distributions for each body part
        predictions_single = {}
        for body_part in body_parts:
            predictions_single[body_part] = get_single_token_distribution(
                model_name="gpt-4.1",
                system_prompt=sys_prompt,
                text_content=prompt_context(text_input),
                body_part=body_part,
                image_path=image_input,
            )

        # Get full output distributions for each body part
        predictions_full = {}
        for body_part in body_parts:
            print(f"Getting full output distribution for {body_part}...")
            predictions_full[body_part] = get_comfort_level_distribution(
                model_name="gpt-4.1",
                system_prompt=sys_prompt,
                text_content=prompt_context(text_input),
                body_part=body_part,
                image_path=image_input,
            )

        # Calculate Brier scores
        brier_scores_full = {}
        brier_scores_single = {}
        brier_scores_comparison = {}

        for body_part in body_parts:
            if body_part in expected_labels:
                # Convert string keys to int for calculation
                label_dict = {
                    int(k): float(v) for k, v in expected_labels[body_part].items()
                }

                brier_scores_full[body_part] = calculate_brier_score(
                    predictions_full[body_part], label_dict
                )
                brier_scores_single[body_part] = calculate_brier_score(
                    predictions_single[body_part], label_dict
                )
                brier_scores_comparison[body_part] = calculate_brier_score(
                    predictions_full[body_part], predictions_single[body_part]
                )

        # Calculate metrics for wrist (primary analysis)
        wrist_single = predictions_single.get("wrist", {})
        wrist_full = predictions_full.get("wrist", {})

        single_token_entropy = calculate_entropy(wrist_single)
        single_token_max_prob = calculate_max_prob(wrist_single)

        full_output_entropy = calculate_entropy(wrist_full)
        full_output_max_prob = calculate_max_prob(wrist_full)

        # Overall Brier scores
        overall_brier_scores = {}
        if "wrist" in expected_labels:
            wrist_labels = {
                int(k): float(v) for k, v in expected_labels["wrist"].items()
            }
            overall_brier_scores["single_token_vs_labels"] = calculate_brier_score(
                wrist_single, wrist_labels
            )
            overall_brier_scores["full_output_vs_labels"] = calculate_brier_score(
                wrist_full, wrist_labels
            )

        # Add result to collection
        all_results[scenario_name] = {
            "scenario": scenario_name,
            "predictions_full": {
                body_part: {str(k): v for k, v in dist.items() if v > 0}
                for body_part, dist in predictions_full.items()
            },
            "predictions_single": {
                body_part: {str(k): v for k, v in dist.items() if v > 0}
                for body_part, dist in predictions_single.items()
            },
            "brier_scores_full": brier_scores_full,
            "brier_scores_single": brier_scores_single,
            "brier_scores_comparison": brier_scores_comparison,
            "labels": expected_labels,
            "single_token": {
                "probs": {str(k): v for k, v in wrist_single.items() if v > 0},
                "entropy": single_token_entropy,
                "max_prob": single_token_max_prob,
                "uncertainty": 1 - single_token_max_prob,
            },
            "full_output": {
                "probs": {str(k): v for k, v in wrist_full.items() if v > 0},
                "entropy": full_output_entropy,
                "max_prob": full_output_max_prob,
                "uncertainty": 1 - full_output_max_prob,
            },
            "brier_scores": overall_brier_scores,
            "comparisons": {
                "brier_single_vs_full": (
                    calculate_brier_score(wrist_full, wrist_single)
                    if (wrist_full and wrist_single)
                    else 0.0
                )
            },
            "expected": {
                "labels": expected_labels,
                "disagreement_type": "verbal_high_face_low_discomfort",
            },
        }

    # Save all results to file
    output_file = "playground/smile_experiment_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print(f"Generated {len(all_results)} scenarios")
