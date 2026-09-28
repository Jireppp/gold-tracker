"""
Check Available Gemini Models
=============================
Queries the official Google GenAI API using GEMINI_API_KEY from .env
and prints all accessible models, their display names, and base model IDs.
"""

import os
import sys

# Ensure UTF-8 console output encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY is not defined in the .env file.")
    sys.exit(1)

print("🔍 Connecting to Google GenAI API and fetching available models...\n")

try:
    client = genai.Client(api_key=api_key)
    models_pager = client.models.list()

    models = list(models_pager)

    if not models:
        print("⚠️ No models returned for this API key.")
        sys.exit(0)

    print(f"Total Accessible Models: {len(models)}")
    print("=" * 80)
    print(f"{'MODEL NAME / ID':<40} | {'DISPLAY NAME':<25} | {'BASE MODEL ID'}")
    print("=" * 80)

    for m in models:
        name = getattr(m, "name", "N/A")
        # Clean prefix if desired (e.g., 'models/')
        clean_name = name.replace("models/", "") if name.startswith("models/") else name
        display_name = getattr(m, "display_name", "N/A") or "N/A"
        base_model_id = getattr(m, "base_model_id", "N/A") or "-"

        print(f"{clean_name:<40} | {display_name:<25} | {base_model_id}")

    print("=" * 80)
    print("\n✅ Listing completed successfully.")

except Exception as err:
    print(f"❌ Failed to list models: {err}")
    sys.exit(1)
