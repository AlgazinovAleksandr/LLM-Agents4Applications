from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import traceback
import logging
from typing import Optional

# Import your agent wrapper
from create_formula import generate_formula

app = FastAPI(
    title="Cosmetics Formula Generator",
    version="0.1",
    description="Accepts a user request and returns a JSON cosmetic formula produced by the agent.",
    docs_url=None,
    redoc_url=None,
)

logger = logging.getLogger("uvicorn.error")


# -----------------------------
# Pydantic models
# -----------------------------
class GenerateRequest(BaseModel):
    message: str
    save_file: bool = False


class GenerateResponse(BaseModel):
    ok: bool
    result: Optional[dict] = None
    error: Optional[dict] = None


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health", tags=["health"])
def health():
    """
    Health check endpoint.
    """
    return {"status": "ok", "message": "Cosmetics microservice is running"}


def format_formula_for_display(formula_data):
    """
    Format the raw formula data into a more readable structure for the frontend.
    """
    if not formula_data or not isinstance(formula_data, dict):
        return formula_data
    
    formatted = {}
    
    # Common cosmetic formula fields to organize (expanded to catch more variations)
    common_fields = {
        "Product Name": ["product_name", "name", "title", "productName", "product name", "product_title"],
        "Description": ["description", "desc", "summary", "overview", "product_description", "product_desc"],
        "Ingredients": ["ingredients", "ingredient_list", "components", "formula", "ingredient", "ingredient_list", "formulation"],
        "Instructions": ["instructions", "usage", "how_to_use", "directions", "application", "instructions_for_use", "how to use", "usage_instructions"],
        "Packaging": ["packaging", "container", "bottle", "tube", "packaging_suggestions", "container_type"],
        "Safety": ["safety", "warnings", "precautions", "regulatory", "safety_information", "safety_and_regulatory", "safety_warnings", "regulatory_information"],
        "Notes": ["notes", "additional_notes", "tips", "recommendations", "additional_information", "additional_notes", "extra_notes", "additional_info"]
    }
    
    # Try to organize the data by common fields
    categorized_keys = set()
    
    for section_name, possible_keys in common_fields.items():
        for key in possible_keys:
            # Look for exact matches first
            actual_key = next((k for k in formula_data.keys() if k.lower() == key.lower()), None)
            if actual_key and actual_key not in categorized_keys:
                if section_name not in formatted:
                    formatted[section_name] = {}
                formatted[section_name][actual_key] = formula_data[actual_key]
                categorized_keys.add(actual_key)
            else:
                # Look for partial matches (key contains the search term)
                for data_key in formula_data.keys():
                    if data_key not in categorized_keys and key.lower() in data_key.lower():
                        if section_name not in formatted:
                            formatted[section_name] = {}
                        formatted[section_name][data_key] = formula_data[data_key]
                        categorized_keys.add(data_key)
                        break
    
    # Add any remaining fields that weren't categorized
    remaining_fields = {k: v for k, v in formula_data.items() if k not in categorized_keys}
    if remaining_fields:
        formatted["Additional Information"] = remaining_fields
    
    # Ensure all expected sections exist, even if empty
    expected_sections = ["Product Name", "Description", "Ingredients", "Instructions", "Packaging", "Safety", "Notes"]
    for section in expected_sections:
        if section not in formatted:
            formatted[section] = {"placeholder": "No information available for this section"}
    
    return formatted if formatted else formula_data

@app.post("/generate", response_model=GenerateResponse, tags=["generation"])
def generate(req: GenerateRequest):
    """
    Generate a cosmetic formula using the agent wrapper.
    Example request body:
      { "message": "Lightweight daytime moisturizer for oily skin with SPF 30", "save_file": false }
    """
    try:
        # Always set save_file=False since we handle downloads in the browser
        result = generate_formula(req.message, save_file=False)
        
        # Format the result for better display
        if result.get("parsed") and result.get("data"):
            formatted_data = format_formula_for_display(result["data"])
            result["formatted_data"] = formatted_data
        
        return GenerateResponse(ok=True, result=result)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("Error while generating formula")
        return GenerateResponse(ok=False, error={"message": str(exc), "trace": tb})


# -----------------------------
# Custom UI now served at root (/). Keep /docs as redirect to /
# -----------------------------
@app.get("/docs", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(url="/")

# Serve the custom UI at the root path (must be last to not interfere with API routes)
app.mount("/", StaticFiles(directory="docs_ui", html=True), name="root")


# -----------------------------
# Run app locally
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
