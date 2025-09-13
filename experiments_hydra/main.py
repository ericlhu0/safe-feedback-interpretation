import glob
import json
import textwrap
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

from safe_feedback_interpretation.models.base_model import BaseModel
from safe_feedback_interpretation.models.openai_model import OpenAIModel


def load_existing_results(results_file: Path) -> set:
    """Load existing experiment results and return set of identifiers to avoid
    duplicates."""
    existing_experiments = set()

    if not results_file.exists():
        return existing_experiments

    try:
        with open(results_file, "r") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line.strip())

                    # Create identifier using config_used (all Hydra parameters), verbal_feedback, and facial_expression
                    config_used = result.get("config_used", {})
                    verbal_feedback = result.get("verbal_feedback", "")
                    facial_expression = result.get("facial_expression", {})
                    facial_description = facial_expression.get("description", "")
                    facial_modality = facial_expression.get("modality", "")

                    # Convert config dict to sorted tuple for hashing
                    config_tuple = tuple(sorted(config_used.items()))
                    identifier = (
                        config_tuple,
                        verbal_feedback,
                        facial_description,
                        facial_modality,
                    )

                    existing_experiments.add(identifier)

    except Exception as e:
        print(f"Warning: Error loading existing results: {e}")

    return existing_experiments


def create_experiment_identifier(
    config_used: dict, verbal_feedback: str, facial_expression: dict
) -> tuple:
    """Create a unique identifier for an experiment combination."""
    facial_description = facial_expression.get("description", "")
    facial_modality = facial_expression.get("modality", "")
    config_tuple = tuple(sorted(config_used.items()))
    print("exp id", config_tuple, verbal_feedback, facial_description, facial_modality)
    return (config_tuple, verbal_feedback, facial_description, facial_modality)


def find_matching_labels(
    scenarios: list, verbal_intensity: str, facial_intensity: str
) -> dict:
    """Find labels from any scenario that matches verbal_intensity and
    facial_intensity, ignoring source_specificity."""
    for scenario in scenarios:
        metadata = scenario.get("experiment_metadata", {})
        if (
            metadata.get("verbal_intensity") == verbal_intensity
            and metadata.get("facial_intensity") == facial_intensity
        ):
            labels = scenario.get("labels", {})
            if labels:  # Return first non-empty labels found
                return labels
    return {}


def generate_feedback_from_metadata(
    verbal_intensity: str,
    facial_intensity: str,
    source_specificity: str,
    face_modality: str,
    assets_dir: Path,
    verbal_data: dict,
    img_to_text_map: dict,
) -> list:
    """Generate all combinations of verbal feedback and facial expressions from
    metadata."""

    # Get verbal feedback options
    verbal_options = verbal_data.get(source_specificity, {}).get(
        verbal_intensity, ["Default feedback"]
    )

    # Get facial expression image files
    facial_dir = assets_dir / "faceimgs" / facial_intensity
    facial_paths = list(facial_dir.glob("*.jpg")) + list(facial_dir.glob("*.png"))

    # Create facial expression options based on modality
    facial_options = []
    for path in facial_paths:
        if face_modality == "img":
            facial_options.append(
                {
                    "modality": "image",
                    "description": "Please examine the provided facial expression image to assess comfort level.",
                    "image_path": str(path),  # Keep path for model input
                }
            )
        else:  # face_modality == "text"
            filename = path.name
            text_description = img_to_text_map.get(
                filename, f"facial expression showing {facial_intensity} intensity"
            )
            facial_options.append({"modality": "text", "description": text_description})

    # Generate combinations
    combinations = []
    for verbal_text in verbal_options:
        for facial_expr in facial_options:
            combinations.append(
                {"verbal_feedback": verbal_text, "facial_expression": facial_expr}
            )

    return combinations


def filter_scenarios_by_metadata(
    scenarios: list,
    verbal_intensity: str,
    facial_intensity: str,
    source_specificity: str,
) -> list:
    """Filter scenarios based on experiment metadata matching Hydra config
    parameters.

    Only matches verbal and facial intensity to allow label reuse across
    source specificities.
    """

    matching_scenarios = []
    for scenario in scenarios:
        metadata = scenario.get("experiment_metadata", {})

        if (
            metadata.get("verbal_intensity") == verbal_intensity
            and metadata.get("facial_intensity") == facial_intensity
        ):
            matching_scenarios.append(scenario)

    return matching_scenarios


def build_model_input(
    base_config: dict,
    verbal_feedback: str,
    facial_expression: dict,
    prompt: str,
) -> str:
    """Build the complete input string for the model."""

    # Extract context from base config
    input_context = base_config["input_context"]

    # Use the facial expression description directly
    facial_desc = facial_expression.get("description", "No facial expression provided")

    # Build context string
    context_str = textwrap.dedent(
        f"""\
        Current action description: {input_context['current_action_description']}
        Current state:
        Contact forces: {input_context['current_state']['contact_forces']}
        Joint angles (deg): {input_context['current_state']['joint_angles_deg']}
        Current comfort threshold:
        Comfort threshold: {input_context['current_comfort_threshold']['current_comfort_threshold']}
        Comfortable joint range (deg): {input_context['current_comfort_threshold']['current_comfortable_joint_range_deg']}
        Received feedback:
        Verbal feedback: {verbal_feedback}
        Facial expression: {facial_desc}
    """
    )

    return context_str + "\n" + prompt


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function that loads Hydra configuration and runs the experiment."""

    # Access configuration parameters
    model_name = cfg.model_name
    face_modality = cfg.face_modality
    verbal_intensity = cfg.verbal_intensity
    facial_intensity = cfg.facial_intensity
    source_specificity = cfg.source_specificity
    body_part = cfg.body_part
    prompt_template = cfg.prompt
    prompt = prompt_template.format(body_part=body_part)
    sys_prompt = cfg.sys_prompt

    print(f"\nConfiguration values:")
    print(f"Model name: {model_name}")
    print(f"Face modality: {face_modality}")
    print(f"Verbal intensity: {verbal_intensity}")
    print(f"Facial intensity: {facial_intensity}")
    print(f"Source specificity: {source_specificity}")
    print(f"Body part: {body_part}")
    print(f"System Prompt: {sys_prompt}\n")
    print(f"Prompt: {prompt}")

    # Instantiate the OpenAI model
    model = OpenAIModel(model=model_name, system_prompt=sys_prompt, temperature=1.0)

    # Load experiment configs from experiments_hydra directory
    labels_dir = Path(__file__).parent / "labels"
    assets_dir = Path(__file__).parent / "assets"
    results_dir = Path(__file__).parent / "results"

    # Load the experiment labels
    config_path = labels_dir / "experiment_labels.json"

    # Load image to text mapping
    img_to_text_path = assets_dir / "img_to_text_map.json"
    img_to_text_map = {}
    try:
        with open(img_to_text_path, "r") as f:
            img_to_text_map = json.load(f)
        print(f"Loaded image-to-text mapping with {len(img_to_text_map)} entries")
    except FileNotFoundError:
        print(f"Image-to-text mapping not found: {img_to_text_path}")
    except Exception as e:
        print(f"Error loading image-to-text mapping: {e}")

    try:
        with open(config_path, "r") as f:
            experiment_config = json.load(f)

        print("Loaded experiment config:")
        print(f"Experiment name: {experiment_config['experiment_name']}")
        print(f"Description: {experiment_config['description']}")

        # Access base config context
        base_config = experiment_config["base_config"]
        input_context = base_config["input_context"]

        print(f"Current action: {input_context['current_action_description']}")
        print(f"Contact forces: {input_context['current_state']['contact_forces']}")
        print(f"Joint angles: {input_context['current_state']['joint_angles_deg']}")

        # Filter scenarios based on Hydra config parameters
        all_scenarios = experiment_config.get("scenarios", [])
        filtered_scenarios = filter_scenarios_by_metadata(
            all_scenarios, verbal_intensity, facial_intensity, source_specificity
        )

        print(f"\nFound {len(all_scenarios)} total scenarios")
        print(f"Filtered to {len(filtered_scenarios)} scenarios matching config:")
        print(
            f"  verbal_intensity={verbal_intensity}, facial_intensity={facial_intensity}, source_specificity={source_specificity}"
        )

        if len(filtered_scenarios) == 0:
            print("No scenarios match the current configuration. Exiting.")
            return

        # Load verbal feedback data for generating combinations
        verbal_data_path = assets_dir / "verbal.json"
        try:
            with open(verbal_data_path, "r") as f:
                verbal_data = json.load(f)
            print(
                f"Loaded verbal feedback data with {len(verbal_data)} source categories"
            )
        except FileNotFoundError:
            print(f"Verbal feedback data not found: {verbal_data_path}")
            return
        except Exception as e:
            print(f"Error loading verbal feedback data: {e}")
            return

        # Set up results file for incremental saving
        results_dir.mkdir(exist_ok=True)  # Ensure results directory exists
        results_file = results_dir / "results.jsonl"
        print(f"Results will be saved to: {results_file}")

        # Load existing results to avoid duplicates
        existing_experiments = load_existing_results(results_file)
        print(f"Found {len(existing_experiments)} existing experiments to skip")

        for i, scenario in enumerate(filtered_scenarios):
            # Create scenario name based on actual config values, not matched scenario
            scenario_name = f"{verbal_intensity}_{facial_intensity}_{source_specificity}_{body_part}"
            print(
                f"\n--- Processing scenario {i+1}/{len(filtered_scenarios)}: {scenario_name} ---"
            )

            # Generate feedback combinations for this scenario
            feedback_combinations = generate_feedback_from_metadata(
                verbal_intensity=verbal_intensity,
                facial_intensity=facial_intensity,
                source_specificity=source_specificity,
                face_modality=face_modality,
                assets_dir=assets_dir,
                verbal_data=verbal_data,
                img_to_text_map=img_to_text_map,
            )

            print(f"Generated {len(feedback_combinations)} feedback combinations")

            for j, feedback_combo in enumerate(feedback_combinations):
                print(
                    f"\n--- Scenario: {scenario_name} | Combination {j+1}/{len(feedback_combinations)} ---"
                )

                verbal_feedback = feedback_combo["verbal_feedback"]
                facial_expression = feedback_combo["facial_expression"]

                # Prepare actual config_used with all Hydra parameters
                actual_config_used = {
                    "model_name": model_name,
                    "face_modality": face_modality,
                    "verbal_intensity": verbal_intensity,
                    "facial_intensity": facial_intensity,
                    "source_specificity": source_specificity,
                    "body_part": body_part,
                }

                # Check if this experiment already exists
                experiment_id = create_experiment_identifier(
                    actual_config_used, verbal_feedback, facial_expression
                )

                if experiment_id in existing_experiments:
                    print(
                        f"Skipping {scenario_name} combination {j+1}/{len(feedback_combinations)} - already exists in results"
                    )
                    continue

                model_input = build_model_input(
                    base_config=base_config,
                    verbal_feedback=verbal_feedback,
                    facial_expression=facial_expression,
                    prompt=prompt,
                )

                print(f"Verbal feedback: {verbal_feedback}")
                print(
                    f"Facial expression: {facial_expression.get('description', 'No description')}"
                )
                print("Model input:")
                print(model_input)

                # Handle image input for facial expressions
                image_path = None
                if (
                    face_modality == "img"
                    and facial_expression.get("modality") == "image"
                ):
                    image_path = facial_expression.get("image_path")
                    print(f"Using image: {image_path}")

                # Call model to get full output
                model_response = None
                error_msg = None
                try:
                    model_response = model.get_full_output(
                        text_input=model_input, image_input=image_path
                    )

                    print(f"\nModel response:")
                    print(model_response)

                except Exception as e:
                    error_msg = str(e)
                    print(f"Error calling model: {e}")

                # Extract labels for this scenario, fallback to matching labels by intensity
                scenario_labels = scenario.get("labels", {})
                if not scenario_labels:
                    scenario_labels = find_matching_labels(
                        all_scenarios, verbal_intensity, facial_intensity
                    )

                # Prepare data to save with actual config values, not from matched scenario
                actual_scenario_metadata = {
                    "verbal_intensity": verbal_intensity,
                    "facial_intensity": facial_intensity,
                    "source_specificity": source_specificity,
                    "body_part": body_part,
                }
                combination_data = {
                    "scenario_name": scenario_name,
                    "scenario_metadata": actual_scenario_metadata,
                    "model_name": model_name,
                    "model_response": model_response,
                    "verbal_feedback": verbal_feedback,
                    "facial_expression": facial_expression,
                    "labels": scenario_labels,
                    "config_used": {
                        "model_name": model_name,
                        "face_modality": face_modality,
                        "verbal_intensity": verbal_intensity,
                        "facial_intensity": facial_intensity,
                        "source_specificity": source_specificity,
                        "body_part": body_part,
                    },
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat(),
                }

                # Append to results file (JSON Lines format)
                try:
                    with open(results_file, "a") as f:
                        f.write(json.dumps(combination_data) + "\n")
                    print(
                        f"Saved combination {j+1}/{len(feedback_combinations)} to results file"
                    )
                except Exception as e:
                    print(f"Error saving combination result: {e}")

            print(
                f"\nCompleted scenario {scenario_name} with {len(feedback_combinations)} combinations"
            )

    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
    except Exception as e:
        print(f"Error loading config: {e}")


if __name__ == "__main__":
    main()
