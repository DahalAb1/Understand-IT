import os
import json
import anthropic
from ...domain.models import ModelOutput, RiskLevel


class AnthropicAdapter:
    max_input_length = 3000

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def simplify(self, text: str) -> ModelOutput:
        message = self.client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a legal document analyzer helping general members of the public "
                        "understand legal contracts without needing a lawyer.\n\n"
                        "Analyze the following clause and respond with a JSON object only, no other text.\n\n"
                        "JSON format:\n"
                        "{\n"
                        '  "title": "name of this legal section (e.g. Indemnification, Termination)",\n'
                        '  "simplified": "plain English explanation preserving legal nuance",\n'
                        '  "risk_level": "low" | "medium" | "high",\n'
                        '  "risk_reason": "why this clause matters to the user, or null if low risk"\n'
                        "}\n\n"
                        f"Clause:\n{text}"
                    ),
                }
            ],
        )

        data = json.loads(message.content[0].text)
        return ModelOutput(
            title=data["title"],
            simplified=data["simplified"],
            risk_level=RiskLevel(data["risk_level"]),
            risk_reason=data.get("risk_reason"),
        )
