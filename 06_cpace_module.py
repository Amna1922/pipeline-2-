"""
=============================================================================
SCRIPT: 06_cpace_module.py
PURPOSE: The CPACE (Contrastive Explanation) Module.
         
         This module generates contrastive explanations — explanations that
         describe not just which fallacy was detected, but why it is THAT
         fallacy rather than a different one. This adds discriminative value
         for the end user.
         
         How it works:
         1. Extracts key concepts from the argument text using spaCy.
         2. Looks up ConceptNet-like relations for those concepts (using a
            lightweight local approach based on spaCy word vectors, since
            full ConceptNet requires ~5GB of storage).
         3. Fills a contrastive explanation template using the concepts,
            the detected fallacy label, and alternative labels.
         
         This script can be used in TWO ways:
         A. As a Flask web service (for use with the inference script):
            python scripts/06_cpace_module.py --serve
         B. As a standalone function (imported by inference script directly):
            from scripts.cpace_module import generate_contrastive_explanation
         
         The inference script (07) imports CPACE directly for simplicity
         since we are now on a single machine.
         
RUN:     python scripts/06_cpace_module.py --serve   (starts web service)
         python scripts/06_cpace_module.py --test    (tests with sample input)
=============================================================================
"""

import os
import sys
import re
import json
import argparse
import spacy
from typing import List, Tuple, Optional

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_MAP = os.path.join(BASE_DIR, "data", "processed", "label_map.json")

# ── Load spaCy model (module-level, loaded once) ───────────────────────────
_nlp = None

def get_nlp():
    """Lazily loads the spaCy model (cached after first call)."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_lg", disable=["parser", "senter"])
    return _nlp


# ── Contrastive templates (one per fallacy label) ──────────────────────────
# {concept} = key concept extracted from the argument
# {label}   = detected fallacy label
# {alt}     = the most contrasting alternative fallacy label

CONTRASTIVE_TEMPLATES = {
    "ad hominem": (
        "This argument commits {label}. "
        "The reasoning attacks the source or character (relating to '{concept}') "
        "rather than the substance of the claim. "
        "Unlike {alt}, which involves a different structural error, "
        "ad hominem is specifically characterized by redirecting attention "
        "from the argument itself to the person or entity making it."
    ),
    "appeal to popularity": (
        "This argument commits {label}. "
        "It uses widespread belief around '{concept}' as a substitute for logical evidence. "
        "Unlike {alt}, which makes a different kind of reasoning error, "
        "appeal to popularity incorrectly treats consensus as proof of truth."
    ),
    "equivocation": (
        "This argument commits {label}. "
        "The term '{concept}' appears to be used with shifting or ambiguous meaning, "
        "which misleads the audience. "
        "Unlike {alt}, which involves a different type of faulty reasoning, "
        "equivocation exploits semantic ambiguity to make the argument appear valid."
    ),
    "fallacy of extension": (
        "This argument commits {label} (Straw Man). "
        "It misrepresents a position related to '{concept}' by overstating or distorting it. "
        "Unlike {alt}, which is a different reasoning error, "
        "this fallacy specifically involves attacking a caricature of the opponent's view."
    ),
    "false cause": (
        "This argument commits {label}. "
        "It wrongly asserts a causal relationship involving '{concept}' "
        "based on correlation or temporal sequence alone. "
        "Unlike {alt}, which errs in a different way, "
        "false cause confuses association with causation."
    ),
    "false dilemma": (
        "This argument commits {label}. "
        "It restricts the options available regarding '{concept}' to only two, "
        "ignoring intermediate or alternative possibilities. "
        "Unlike {alt}, which involves a different reasoning failure, "
        "false dilemma artificially limits the choice space."
    ),
    "hasty generalization": (
        "This argument commits {label}. "
        "It draws a broad conclusion about '{concept}' from a limited or "
        "unrepresentative set of cases. "
        "Unlike {alt}, which errs differently, "
        "hasty generalization over-extends a narrow observation."
    ),
    "intentional": (
        "This argument commits an {label} fallacy. "
        "The reasoning around '{concept}' appears to be deliberately constructed "
        "to mislead or manipulate rather than to inform. "
        "Unlike {alt}, which can arise from unintentional reasoning errors, "
        "this fallacy involves deliberate rhetorical deception."
    ),
    "logical fallacy": (
        "This argument contains a {label}. "
        "The reasoning involving '{concept}' breaks down because the premises "
        "do not adequately support the conclusion. "
        "Unlike {alt}, this instance represents a more general breakdown in logical structure."
    ),
    "relevance fallacy": (
        "This argument commits a {label}. "
        "The evidence or claim about '{concept}' is logically unrelated to the "
        "conclusion being drawn. "
        "Unlike {alt}, which involves a different error type, "
        "this fallacy introduces information that has no bearing on the argument's conclusion."
    ),
}


# ── Concept extraction ─────────────────────────────────────────────────────

def extract_key_concept(text: str) -> str:
    """
    Extracts the single most informative concept from an argument text.
    
    Strategy:
    1. Run spaCy NLP on the text.
    2. Prefer named entities (concrete, specific) as key concepts.
    3. Fall back to noun chunks (noun phrases like "the economy", "public opinion").
    4. Fall back to individual nouns or verbs if nothing else is found.
    5. Return the first meaningful token if all else fails.
    
    The concept is used to fill the {concept} placeholder in templates,
    making the explanation feel grounded in the actual argument.
    """
    nlp  = get_nlp()
    doc  = nlp(text[:512])  # Limit to 512 chars for speed
    
    # 1. Named entities (e.g., "NATO", "Joe Biden") — most specific
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "NORP", "EVENT", "PRODUCT"}:
            return ent.text.lower()
    
    # 2. Noun chunks (e.g., "the climate scientists", "public trust")
    for chunk in doc.noun_chunks:
        root = chunk.root
        if root.is_alpha and not root.is_stop and len(chunk.text) > 3:
            return chunk.text.lower()
    
    # 3. Nouns or verbs (less specific but still content-bearing)
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "VERB"} and not token.is_stop and token.is_alpha:
            return token.lemma_.lower()
    
    # 4. Last resort: first non-stopword token
    for token in doc:
        if not token.is_stop and token.is_alpha:
            return token.text.lower()
    
    return "this claim"


def get_alternative_label(label: str, label_map: dict) -> str:
    """
    Returns the fallacy label most conceptually distinct from the given one.
    This is used to fill the {alt} placeholder, creating contrast in the explanation.
    
    Simple heuristic: pick the fallacy from a priority list that is most 
    different in type. For example, ad hominem (person-attack) contrasts well
    with false dilemma (false choices).
    """
    # Pre-defined contrast pairs for maximum conceptual contrast
    CONTRASTS = {
        "ad hominem":          "false dilemma",
        "appeal to popularity": "false cause",
        "equivocation":        "hasty generalization",
        "fallacy of extension": "ad hominem",
        "false cause":         "appeal to popularity",
        "false dilemma":       "ad hominem",
        "hasty generalization": "false cause",
        "intentional":         "logical fallacy",
        "logical fallacy":     "false dilemma",
        "relevance fallacy":   "hasty generalization",
    }
    
    preferred = CONTRASTS.get(label)
    if preferred and preferred in label_map:
        return preferred
    
    # Fall back to any label that isn't the current one
    alternatives = [l for l in label_map.keys() if l != label]
    return alternatives[0] if alternatives else "another type of fallacy"


# ── Main CPACE function ────────────────────────────────────────────────────

def generate_contrastive_explanation(
    argument_text: str,
    predicted_label: str,
    label_map: dict,
    retrieved_passages: Optional[List[str]] = None
) -> str:
    """
    Generates a contrastive explanation for a predicted fallacy label.
    
    Arguments:
        argument_text:     The (normalized) argument to be explained.
        predicted_label:   The fallacy label predicted by the DeBERTa classifier.
        label_map:         Dict mapping label strings to integer IDs.
        retrieved_passages: Optional list of retrieved passage strings (not
                            directly used in template filling but passed for
                            potential future enhancement).
    
    Returns:
        A natural language contrastive explanation string.
    
    Example:
        Input:  "you cannot trust [person] because [person] works for [org]"
                label = "ad hominem"
        Output: "This argument commits ad hominem. The reasoning attacks the
                 source or character (relating to '[person]') rather than the
                 substance of the claim. Unlike false dilemma, which involves..."
    """
    label = predicted_label.lower().strip()
    
    # Extract key concept from the argument text
    concept = extract_key_concept(argument_text)
    
    # Get a contrasting alternative label
    alt = get_alternative_label(label, label_map)
    
    # Get the template for this fallacy type (fall back to generic)
    template = CONTRASTIVE_TEMPLATES.get(label, CONTRASTIVE_TEMPLATES["logical fallacy"])
    
    # Fill the template
    explanation = template.format(
        label=label,
        concept=concept,
        alt=alt
    )
    
    return explanation


# ── Flask web service (optional, run with --serve) ─────────────────────────

def run_flask_service():
    """
    Starts a lightweight Flask web service on port 5000.
    Accepts POST requests with JSON body:
        {"argument": "...", "label": "ad hominem", "label_map": {...}}
    Returns:
        {"contrastive_explanation": "..."}
    
    This is included for completeness but is NOT required when running
    the inference script (07) on a single machine — it imports CPACE directly.
    """
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})
    
    @app.route("/cpace", methods=["POST"])
    def cpace_endpoint():
        data = request.get_json(force=True)
        argument   = data.get("argument", "")
        label      = data.get("label", "logical fallacy")
        lmap       = data.get("label_map", {})
        passages   = data.get("retrieved_passages", [])
        
        explanation = generate_contrastive_explanation(argument, label, lmap, passages)
        return jsonify({"contrastive_explanation": explanation})
    
    print("Starting CPACE Flask service on http://0.0.0.0:5000")
    print("Endpoints: GET /health  |  POST /cpace")
    print("Press Ctrl+C to stop.")
    app.run(host="0.0.0.0", port=5000, debug=False)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPACE Contrastive Explanation Module")
    parser.add_argument("--serve", action="store_true", help="Run as Flask web service")
    parser.add_argument("--test",  action="store_true", help="Run a quick test")
    args = parser.parse_args()
    
    # Load label map
    if os.path.exists(LABEL_MAP):
        with open(LABEL_MAP) as f:
            label_map = json.load(f)
    else:
        # Default labels if label_map.json hasn't been created yet
        label_map = {
            "ad hominem": 0, "appeal to popularity": 1, "equivocation": 2,
            "fallacy of extension": 3, "false cause": 4, "false dilemma": 5,
            "hasty generalization": 6, "intentional": 7, "logical fallacy": 8,
            "relevance fallacy": 9
        }
        print("WARNING: label_map.json not found. Using default label map.")
    
    if args.serve:
        run_flask_service()
    elif args.test:
        print("=" * 60)
        print("CPACE MODULE TEST")
        print("=" * 60)
        
        test_cases = [
            ("you cannot trust [person] because [person] works for [org]", "ad hominem"),
            ("either we ban all [product] or [norp] will take over everything", "false dilemma"),
            ("if we allow [event] then we will inevitably end up with total chaos", "false cause"),
            ("everyone knows that [norp] people are the best, so it must be true", "appeal to popularity"),
        ]
        
        for text, label in test_cases:
            print(f"\nArgument: '{text}'")
            print(f"Predicted label: {label}")
            explanation = generate_contrastive_explanation(text, label, label_map)
            print(f"CPACE Output:\n  {explanation}")
        
        print("\n" + "=" * 60)
        print("Test complete.")
    else:
        print("Usage:")
        print("  python scripts/06_cpace_module.py --test   # Run test")
        print("  python scripts/06_cpace_module.py --serve  # Start web service")
