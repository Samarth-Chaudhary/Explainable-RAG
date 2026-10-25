"""Prompt Injection Defense, Instruction Neutralization, and Data Isolation Service.

Replaces naive keyword blocklists with a multi-layered defense-in-depth architecture:
1. Canonicalization: NFKC Unicode normalization, zero-width stripping, and homoglyph normalization.
2. Structural Data Isolation: Wrapping untrusted text in cryptographically nonced XML tags.
3. Instruction Neutralization: Stripping LLM chat tokens ([INST], <<SYS>>, <|im_start|>) and escaping markdown/XML.
4. Multi-Vector Detection: Pattern detection across direct overrides, leetspeak, base64/hex encoding,
   multilingual variants (Spanish, French, German, Russian, Chinese, Hindi, Arabic), role hijacking,
   and exfiltration directives.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import re
import unicodedata
import uuid

# Zero-width and invisible control codepoints
ZERO_WIDTH_CHARS = {
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # zero-width no-break space (BOM)
    "\u2060",  # word joiner
    "\u00ad",  # soft hyphen
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\u202a",  # left-to-right embedding
    "\u202b",  # right-to-left embedding
    "\u202c",  # pop directional formatting
    "\u202d",  # left-to-right override
    "\u202e",  # right-to-left override
}

# Common Cyrillic and Greek lookalike homoglyphs mapped to Latin ASCII
HOMOGLYPH_MAP = {
    "а": "a", "А": "A", "с": "c", "С": "C", "е": "e", "Е": "E",
    "о": "o", "О": "O", "р": "p", "Р": "P", "х": "x", "Х": "X",
    "у": "y", "У": "Y", "і": "i", "І": "I", "ј": "j", "Ј": "J",
    "ѕ": "s", "Ѕ": "S", "ԁ": "d", "Ԃ": "D", "ԛ": "q", "Ԝ": "w",
    "α": "a", "β": "b", "γ": "g", "ε": "e", "η": "h", "ι": "i",
    "κ": "k", "ν": "v", "ο": "o", "ρ": "p", "τ": "t", "υ": "u",
    "χ": "x",
}

# Regex for stripping LLM special tokens
CHAT_TEMPLATE_TOKENS = re.compile(
    r"\[/?INST\]|<<SYS>>|<</SYS>>|<\|im_start\|>|<\|im_end\|>|</?system>",
    re.IGNORECASE,
)

# Multi-vector attack signatures
ATTACK_RULES: list[tuple[str, str, re.Pattern]] = [
    # Direct system prompt overrides
    (
        "direct_override",
        "Direct instruction override",
        re.compile(
            r"(ignore|disregard|forget|bypass|override)\s+(all\s+)?(previous|prior|above|system)?\s*(instructions|prompts?|rules?|directives?)",
            re.IGNORECASE,
        ),
    ),
    (
        "jailbreak_mode",
        "Jailbreak or persona adoption",
        re.compile(
            r"\b(DAN\s+mode|developer\s+mode|jailbreak|unrestricted\s+mode|do\s+anything\s+now)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_hijacking",
        "Chat template or role spoofing",
        re.compile(
            r"(^|\n)\s*(system|assistant|user|human|reviewer)\s*:\s*(ignore|approve|override|you\s+must)",
            re.IGNORECASE,
        ),
    ),
    (
        "chat_template_injection",
        "LLM chat template control tokens",
        re.compile(
            r"\[/?INST\]|<<SYS>>|<</SYS>>|<\|im_start\|>|<\|im_end\|>",
            re.IGNORECASE,
        ),
    ),
    # Multilingual overrides
    (
        "multilingual_spanish",
        "Spanish instruction override",
        re.compile(
            r"\b(ignora|olvida|anula)\s+(todas\s+)?(las\s+)?instrucciones\s+(anteriores|previas)?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "multilingual_french",
        "French instruction override",
        re.compile(
            r"\b(ignorez?|oubliez?)\s+(toutes\s+)?(les\s+)?instructions\s+pr[ée]c[ée]dentes\b",
            re.IGNORECASE,
        ),
    ),
    (
        "multilingual_german",
        "German instruction override",
        re.compile(
            r"\b(ignoriere|vergesse?)\s+(alle\s+)?vorherigen?\s+anweisungen\b",
            re.IGNORECASE,
        ),
    ),
    (
        "multilingual_russian",
        "Russian instruction override",
        re.compile(
            r"(игнорируй(те)?|забудь(те)?)\s+(все\s+)?(предыдущие\s+)?инструкции",
            re.IGNORECASE,
        ),
    ),
    (
        "multilingual_chinese",
        "Chinese instruction override",
        re.compile(
            r"忽略(之前|前面|历史)?(的)?(所有)?(指令|提示|指示|规则)|重置(所有)?指令",
        ),
    ),
    (
        "multilingual_hindi",
        "Hindi instruction override",
        re.compile(
            r"(पिछल[ेा]|सभी)\s+निर्देश(ों)?\s+को\s+अनदेखा\s+कर[ेंो]|सिस्टम\s+निर्देश\s+भूल\s+जाओ",
        ),
    ),
    (
        "multilingual_arabic",
        "Arabic instruction override",
        re.compile(
            r"(تجاهل|انسى)\s+(جميع\s+)?(ال)?تعليمات\s+السابقة",
        ),
    ),
    # Obfuscated / Leetspeak instruction overrides
    (
        "obfuscated_leetspeak",
        "Leetspeak instruction override",
        re.compile(
            r"\b[1i][g9][nñ][0o]r[3e]\b.*\b(pr[0o]mpt|[1i]nstruct[1i][0o]n|[0o]v[3e]rr[1i]d[3e])\b",
            re.IGNORECASE,
        ),
    ),
    (
        "spaced_characters",
        "Spaced character obfuscation",
        re.compile(
            r"i\s+g\s+n\s+o\s+r\s+e\s+(p\s+r\s+e\s+v\s+i\s+o\s+u\s+s\s+)?i\s+n\s+s\s+t\s+r\s+u\s+c\s+t\s+i\s+o\s+n\s+s",
            re.IGNORECASE,
        ),
    ),
    # Exfiltration directives
    (
        "data_exfiltration",
        "System prompt or credential exfiltration",
        re.compile(
            r"(reveal|print|output|dump|leak|show|exfiltrate)\s+(the\s+)?(system\s+prompt|api[_\s-]?keys?|database\s+credentials?|passwords?|secret)",
            re.IGNORECASE,
        ),
    ),
    # Delimiter breakout
    (
        "delimiter_breakout",
        "XML or Markdown context breakout",
        re.compile(
            r"</untrusted_policy_data>|===+\s*END\s+(OF\s+)?(SYSTEM\s+)?CONTEXT\s*===+|```+\s*(system|assistant)",
            re.IGNORECASE,
        ),
    ),
]


@dataclass
class InjectionScanResult:
    """Detailed forensic result of prompt injection scan."""

    is_safe: bool
    attack_category: str | None
    risk_score: float  # 0.0 (safe) to 1.0 (malicious)
    matched_rules: list[str]
    neutralized_content: str
    warning_message: str | None = None


class PromptInjectionDefenseService:
    """Multi-stage security service for prompt injection detection and neutralization."""

    def __init__(self, high_risk_threshold: float = 0.5):
        self.high_risk_threshold = high_risk_threshold

    def canonicalize(self, text: str) -> str:
        """Apply Unicode NFKC normalization, strip zero-width chars, and translate homoglyphs."""
        # 1. NFKC normalization
        norm = unicodedata.normalize("NFKC", text)

        # 2. Strip invisible and zero-width codepoints
        cleaned_chars = [c for c in norm if c not in ZERO_WIDTH_CHARS]
        cleaned = "".join(cleaned_chars)

        # 3. Translate homoglyphs to Latin equivalents
        homoglyph_translated = "".join(HOMOGLYPH_MAP.get(c, c) for c in cleaned)
        return homoglyph_translated

    def neutralize(self, text: str) -> str:
        """Strip dangerous instruction tokens, escape markdown fences and XML delimiters."""
        canonical = self.canonicalize(text)

        # Strip chat template tokens ([INST], <<SYS>>, <|im_start|>)
        stripped = CHAT_TEMPLATE_TOKENS.sub(" ", canonical)

        # Escape markdown fences to prevent delimiter breakouts
        escaped_md = stripped.replace("```", r"\`\`\`")

        # Escape custom containment XML tags
        escaped_xml = escaped_md.replace("</untrusted_policy_data>", "&lt;/untrusted_policy_data&gt;")
        escaped_xml = escaped_xml.replace("<untrusted_policy_data", "&lt;untrusted_policy_data")

        return escaped_xml.strip()

    def wrap_as_passive_data(self, text: str, nonce: str | None = None) -> str:
        """Wrap untrusted text strictly as passive data inside cryptographically nonced tags."""
        safe_nonce = nonce or uuid.uuid4().hex[:12]
        neutralized_text = self.neutralize(text)
        return (
            f'<untrusted_policy_data nonce="{safe_nonce}">\n'
            f"{neutralized_text}\n"
            f"</untrusted_policy_data>"
        )

    def _check_encoded_payloads(self, text: str) -> list[str]:
        """Inspect Base64 or Hex chunks within text for hidden injection directives."""
        findings: list[str] = []

        # Find potential base64 strings (>= 16 characters, alphanumeric with +/=)
        b64_matches = re.findall(r"\b[A-Za-z0-9+/]{16,}={0,2}\b", text)
        for b64 in b64_matches:
            try:
                decoded_bytes = base64.b64decode(b64.encode("ascii"), validate=True)
                decoded_text = decoded_bytes.decode("utf-8", errors="ignore").lower()
                if any(k in decoded_text for k in ["ignore", "system", "override", "instruction", "prompt", "secret", "password"]):
                    findings.append("base64_encoded_injection")
            except Exception:
                pass

        # Find hex escapes (\x49\x67...)
        hex_matches = re.findall(r"(?:\\x[0-9a-fA-F]{2}){4,}", text)
        for hx in hex_matches:
            try:
                raw_bytes = bytes.fromhex(hx.replace("\\x", ""))
                decoded_text = raw_bytes.decode("utf-8", errors="ignore").lower()
                if any(k in decoded_text for k in ["ignore", "system", "override", "instruction"]):
                    findings.append("hex_encoded_injection")
            except Exception:
                pass

        return findings

    def scan(self, text: str) -> InjectionScanResult:
        """Execute comprehensive forensic scan across text, decoded payloads, and leetspeak."""
        if not text or not text.strip():
            return InjectionScanResult(
                is_safe=True,
                attack_category=None,
                risk_score=0.0,
                matched_rules=[],
                neutralized_content="",
            )

        canonical_text = self.canonicalize(text)
        neutralized_text = self.neutralize(text)
        matched_rules: list[str] = []
        categories: list[str] = []

        # 1. Evaluate pattern rules against canonical and raw text
        for rule_id, category, pattern in ATTACK_RULES:
            if pattern.search(canonical_text) or pattern.search(text):
                matched_rules.append(rule_id)
                categories.append(category)

        # 2. Evaluate encoded payloads (base64 / hex)
        encoded_findings = self._check_encoded_payloads(text)
        if encoded_findings:
            matched_rules.extend(encoded_findings)
            categories.append("Encoded instruction payload")

        # 3. Calculate risk score
        if not matched_rules:
            risk_score = 0.0
            is_safe = True
            primary_category = None
            warning = None
        else:
            risk_score = min(1.0, 0.4 + 0.3 * len(matched_rules))
            is_safe = risk_score < self.high_risk_threshold
            primary_category = categories[0] if categories else "Unknown attack"
            warning = f"Suspected prompt injection detected: {primary_category} (Rules: {', '.join(matched_rules)})."

        return InjectionScanResult(
            is_safe=is_safe,
            attack_category=primary_category,
            risk_score=risk_score,
            matched_rules=matched_rules,
            neutralized_content=neutralized_text,
            warning_message=warning,
        )
