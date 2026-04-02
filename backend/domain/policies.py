from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClausePolicy:
    clause_type: str
    risk_level: str
    base_reason: str
    review_triggers: tuple[str, ...] = ()
    required_focus: tuple[str, ...] = ()
    review_questions: tuple[str, ...] = ()
    precision_note: str | None = None


CLAUSE_POLICIES: dict[str, ClausePolicy] = {
    "indemnity": ClausePolicy(
        clause_type="indemnity",
        risk_level="high",
        base_reason="Indemnity clauses can shift legal and financial responsibility in a broad way.",
        review_triggers=("defend", "hold harmless", "third party", "losses", "claims"),
        required_focus=("scope of covered claims", "who pays defense costs", "whether the duty is capped"),
        review_questions=(
            "Does this clause force one side to pay for third-party claims or legal defense costs?",
            "Is the indemnity obligation limited, mutual, or one-sided?",
        ),
        precision_note="Check who is protected, what claims are covered, and whether defense costs are included.",
    ),
    "liability": ClausePolicy(
        clause_type="liability",
        risk_level="high",
        base_reason="Liability clauses can cap damages, exclude remedies, or leave one side exposed.",
        review_triggers=("consequential", "indirect", "special damages", "aggregate liability", "cap"),
        required_focus=("damage exclusions", "liability cap", "exceptions to the cap"),
        review_questions=(
            "Does this clause cap liability at a level that would still protect you?",
            "Are there important exceptions that remove the cap for some claims?",
        ),
        precision_note="Check excluded damages, total liability caps, and any carve-outs.",
    ),
    "termination": ClausePolicy(
        clause_type="termination",
        risk_level="high",
        base_reason="Termination clauses control how and when the relationship can end.",
        review_triggers=("for cause", "without cause", "notice", "immediately", "survive"),
        required_focus=("termination triggers", "notice period", "post-termination duties"),
        review_questions=(
            "Can the other side terminate easily while you remain locked in?",
            "What obligations survive after termination?",
        ),
        precision_note="Check the notice period, cause standards, and what continues after termination.",
    ),
    "arbitration": ClausePolicy(
        clause_type="arbitration",
        risk_level="high",
        base_reason="Arbitration clauses can waive court access and change dispute rights.",
        review_triggers=("binding arbitration", "jury trial", "class action", "venue"),
        required_focus=("dispute forum", "waivers", "cost of enforcement"),
        review_questions=(
            "Does this clause waive the right to sue in court or join a class action?",
            "Is the dispute forum practical and affordable?",
        ),
        precision_note="Check whether court rights, jury rights, or class-action rights are waived.",
    ),
    "ip": ClausePolicy(
        clause_type="ip",
        risk_level="high",
        base_reason="IP clauses can transfer ownership, limit reuse, or create licensing restrictions.",
        review_triggers=("assign", "work made for hire", "license", "ownership", "derivative works"),
        required_focus=("ownership transfer", "license scope", "reuse rights"),
        review_questions=(
            "Does this clause transfer ownership of work product or intellectual property?",
            "Are any retained rights or license back rights clearly stated?",
        ),
        precision_note="Check who owns new work, prior materials, and any ongoing license rights.",
    ),
    "renewal": ClausePolicy(
        clause_type="renewal",
        risk_level="medium",
        base_reason="Renewal clauses can extend obligations automatically if deadlines are missed.",
        review_triggers=("automatically renew", "renewal term", "notice of non-renewal"),
        required_focus=("renewal trigger", "notice window", "term length"),
        review_questions=(
            "Will this agreement renew automatically if no one acts in time?",
            "Is the non-renewal notice window realistic?",
        ),
        precision_note="Check renewal timing, notice deadlines, and whether price changes can happen on renewal.",
    ),
    "confidentiality": ClausePolicy(
        clause_type="confidentiality",
        risk_level="medium",
        base_reason="Confidentiality clauses can create long-lasting use and disclosure restrictions.",
        review_triggers=("confidential information", "use restriction", "disclose", "return or destroy"),
        required_focus=("scope of confidential information", "permitted disclosures", "duration"),
        review_questions=(
            "Is confidential information defined too broadly?",
            "Are the allowed disclosures and duration reasonable?",
        ),
        precision_note="Check what counts as confidential, allowed exceptions, and how long duties last.",
    ),
    "payment": ClausePolicy(
        clause_type="payment",
        risk_level="medium",
        base_reason="Payment clauses can create fees, penalties, or timing obligations that materially affect cost.",
        review_triggers=("invoice", "late fee", "interest", "non-refundable", "payment due"),
        required_focus=("payment trigger", "timing", "penalties"),
        review_questions=(
            "When is payment due, and what happens if payment is late?",
            "Are there non-refundable fees, penalties, or automatic increases?",
        ),
        precision_note="Check due dates, late fees, refund limits, and any price escalation terms.",
    ),
}


def get_clause_policy(clause_type: str) -> ClausePolicy | None:
    return CLAUSE_POLICIES.get(clause_type)
