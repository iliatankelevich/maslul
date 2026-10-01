"""Live probe against the real TypeSafe Jev API — one Hebrew classification, one Hebrew
verification. Costs a fraction of a cent. Set TYPESAFE_API_KEY (or pass api_key_env=), then:

uv run --extra jev python examples/jev_probe.py
"""

from __future__ import annotations

import asyncio

from maslul.jev import JevClassifier, JevVerifier
from maslul.types import Message, Request, Response, Usage


async def main() -> None:
    classifier = JevClassifier()
    verifier = JevVerifier()

    simple = Request(messages=[Message(role="user", content="תודה קיפי!")])
    hard = Request(
        messages=[
            Message(
                role="user",
                content=(
                    "אני צריך לתכנן מעבר דירה לעיר אחרת תוך חודש: למצוא דירה חדשה, "
                    "לתאם הובלה, לבטל ולפתוח חוזים (חשמל, אינטרנט, ביטוח), לעדכן כתובת "
                    "במוסדות, ולסדר רישום בית ספר לילדים — תבנה לי לוח זמנים מסודר."
                ),
            )
        ]
    )

    for label, req in [("simple", simple), ("hard", hard)]:
        decision = await classifier.decide(req)
        print(
            f"[classify:{label}] level={decision.level} confidence={decision.confidence} "
            f"probabilities={decision.probabilities} model={decision.model} usage={decision.usage}"
        )

    reply = Response(
        text="בשמחה!",
        level_used=None,
        provider="probe",
        model="probe",
        usage=Usage(),
    )
    verdict = await verifier.judge(simple, reply)
    print(f"[verify] p_yes={verdict.p_yes} model={verdict.model} usage={verdict.usage}")


if __name__ == "__main__":
    asyncio.run(main())
