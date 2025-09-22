"""
Wrapper module that exposes `generate_formula(message: str, save_file: bool=False)`.
This module is written to match the Agent usage that FastAPI app imports.

Notes:
- This expects a module `constants.py` with variables: model_name, base_url, api_key, or .env file with these variables.
- This expects the `autogen` package (or whatever package provides
  AssistantAgent and UserProxyAgent) to be installable/importable.
- If your actual agent API differs slightly, adjust `initiate_chat` and
  how the chat result is read accordingly.
"""

import re
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --- Try to import Agent classes and constants, give helpful errors if missing ---
try:
    # Replace with the actual package that provides these classes if different
    from autogen import AssistantAgent, UserProxyAgent
except Exception as e:
    raise ImportError(
        "Failed to import `AssistantAgent` and `UserProxyAgent` from `autogen`.\n"
        "Make sure the package that provides these classes is installed and available.\n"
        "If the package name differs, update run_cosmetics.py accordingly.\n"
        f"Original error: {e}"
    ) from e

try:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    model_name = os.getenv("model_name")
    base_url = os.getenv("base_url")
    api_key = os.getenv("api_key")
except Exception as e:
    raise ImportError(
        "Failed to import `constants`. Retrieve base_url, model_name, and api_key either from constants.py or .env.\n"
        f"Original error: {e}"
    ) from e


# --- Helper: robustly extract / parse JSON-like output from the agent ---
def _extract_json_like(text: str) -> str:
    """
    Attempt to extract a JSON substring from `text`.
    1) Remove a trailing 'TERMINATE' marker if present.
    2) Find the first '{' and the last '}' and return that slice.
    3) If not found, return the original cleaned text (so caller may decide).
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    # Remove TERMINATE marker and common separators
    cleaned = re.sub(r'\bTERMINATE\b', '', text, flags=re.IGNORECASE).strip()

    # Try to find a JSON object inside the cleaned text
    first_curly = cleaned.find('{')
    last_curly = cleaned.rfind('}')
    if first_curly != -1 and last_curly != -1 and last_curly > first_curly:
        return cleaned[first_curly:last_curly + 1]

    # As a fallback, try to find a JSON array
    first_brack = cleaned.find('[')
    last_brack = cleaned.rfind(']')
    if first_brack != -1 and last_brack != -1 and last_brack > first_brack:
        return cleaned[first_brack:last_brack + 1]

    # Nothing obvious found — return the cleaned original
    return cleaned


def _parse_json_string(json_like_str: str) -> Any:
    """
    Try parsing the provided string as JSON. Raise on failure.
    """
    # Final cleanup: remove excessive newlines that can break naive parsers
    candidate = json_like_str.strip()
    # Replace occurrences of single quotes with double quotes only as a last-ditch attempt
    # (not ideal for all cases, but sometimes the agent returns Python-like dicts)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Try the bracket/curly extraction then parse
        extracted = _extract_json_like(candidate)
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            # Try naive single-quote -> double-quote replacement to handle python dict style
            alt = re.sub(r"'", '"', extracted)
            try:
                return json.loads(alt)
            except json.JSONDecodeError as e:
                # Give up and propagate more informative error
                raise ValueError(
                    "Failed to parse JSON-like string. "
                    "Tried raw, extracted JSON-like parts, and single-quote replacement."
                ) from e


# --- Lazy initialization of the agent instances ---
_agent = None
_user_proxy = None


def _init_agents():
    """
    Initialize AssistantAgent and UserProxyAgent instances (lazy).
    Adjust the llm_config or system prompt as needed for your setup.
    """
    global _agent, _user_proxy
    if _agent is not None and _user_proxy is not None:
        return

    # Build LLM config from constants
    llm_config = {"config_list": [{
            "model": model_name,  # or other supported model IDs
            "base_url": base_url,
            "api_key": api_key,
        }]}

    system_prompt_cosmetics = """You will get an idea from the user's objectives and create a cosmetic product formula that satisfies the objectives. The formula should include the following sections:
1. Product Name: A catchy and relevant name for the cosmetic product.
2. Description: A brief description of the product, its benefits, and its target audience.
3. Ingredients: A detailed list of ingredients, including their functions and concentrations.
4. Instructions for Use: Clear and concise instructions on how to use the product.
5. Packaging: Suggestions for packaging that aligns with the product's branding and target audience.
6. Safety and Regulatory Information: Any necessary safety warnings or regulatory information.
7. Additional Notes: Any other relevant information or tips for the user.
Return the formula in a structured format, preferably valid JSON. After the formula is created, return the word "TERMINATE" to end the conversation.
"""

    # Create agents (names and args can be adapted depending on your autogen API)
    _agent = AssistantAgent(
        name="Formula_Creator",
        system_message=system_prompt_cosmetics,
        llm_config=llm_config,
    )

    _user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",  # never expect interactive human input here
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
        code_execution_config={"work_dir": "coding", "use_docker": False},
    )


# --- Public API: generate_formula ---
def generate_formula(user_message: str, save_file: bool = False, save_name='formula.json') -> Dict[str, Any]:
    """
    Run the agent flow to generate a cosmetic formula from `user_message`.

    Returns a dictionary with:
      {
        "parsed": bool,       # whether parsing to JSON succeeded
        "data": dict|list|None,# parsed JSON (if parsed True)
        "raw": str,           # raw text returned by the agent
        "error": Optional[str] # error message if something went wrong
      }
    """
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("`user_message` must be a non-empty string.")

    _init_agents()

    try:
        # Initiate chat - this call and the returned structure depend on your autogen API.
        chat_result = _user_proxy.initiate_chat(_agent, message=user_message)

        # Attempt to read the final assistant message content.
        # This assumes `chat_result.chat_history` is a list of dicts with 'content' keys.
        # If your autogen returns a different shape, adapt here.
        last_content = None
        if hasattr(chat_result, "chat_history"):
            history = chat_result.chat_history
            if isinstance(history, list) and history:
                last_content = history[-1].get("content")
        # Fallbacks if chat_result provides different access patterns:
        if last_content is None and isinstance(chat_result, dict):
            # some APIs return {'chat_history': [...]}
            hist = chat_result.get("chat_history") or chat_result.get("history")
            if isinstance(hist, list) and hist:
                last_content = hist[-1].get("content")

        if not last_content:
            # If we didn't find the final assistant message, try common attributes
            last_content = getattr(chat_result, "content", None) or str(chat_result)

        raw = last_content if isinstance(last_content, str) else str(last_content)

        # Try to parse JSON-like content
        try:
            json_like = _extract_json_like(raw)
            parsed = _parse_json_string(json_like)
            result = {"parsed": True, "data": parsed, "raw": raw, "error": None}
            if save_file:
                # write to disk using safe filename
                try:
                    with open(save_name, "w", encoding="utf-8") as f:
                        json.dump(parsed, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.warning("Failed saving parsed result to formula.json: %s", e)
            return result
        except Exception as parse_exc:
            # Parsing failed, return raw output and parsing error (but keep the raw text)
            logger.exception("Failed to parse agent output as JSON.")
            return {"parsed": False, "data": None, "raw": raw, "error": str(parse_exc)}

    except Exception as exc:
        logger.exception("Error while running the agent.")
        return {"parsed": False, "data": None, "raw": "", "error": str(exc)}


# Allow running this file directly for quick local tests (will raise if autogen/constants missing)
if __name__ == "__main__":
    import sys

    example = "I want a lightweight daytime moisturizer for oily skin with SPF 30"
    if len(sys.argv) > 1:
        example = " ".join(sys.argv[1:])

    print("Running generate_formula with message:", example)
    out = generate_formula(example, save_file=False, save_name='formula1.json')
    print("Result:", json.dumps(out, indent=2, ensure_ascii=False))
