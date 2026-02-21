import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def make_prompt(representative_docs, level="top"):
    joined = "\n\n---\n\n".join([d.strip()[:700] for d in representative_docs])

    if level == "top":
        return (
            "Label this cluster of 20 Newsgroups posts.\n"
            "Pick a short label and a 1-sentence description.\n\n"
            f"Documents:\n{joined}\n"
        )

    return (
        "Label this sub-cluster inside a bigger topic.\n"
        "Pick a short label and a 1-sentence description.\n\n"
        f"Documents:\n{joined}\n"
    )

def label_with_openai(prompt: str):
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Force strict JSON output using a schema
    resp = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "cluster_label",
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "2-5 word topic label"
                        },
                        "description": {
                            "type": "string",
                            "description": "Exactly one sentence"
                        }
                    },
                    "required": ["label", "description"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    )

    # The Responses API returns JSON as text; we can parse via Python json
    import json
    return json.loads(resp.output_text)