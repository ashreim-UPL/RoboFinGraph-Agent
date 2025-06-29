# report_summaries.py

report_section_specs = {
    "company_overview.txt": {
        "sources": ["analyze_company_description", "analyze_business_highlights"],
        "word_limit": 130,
        "prompt_template": (
            "IMPORTANT: Reply using only plain text. Do not use any markdown formatting, bold, italics, underlines, bullets, numbers, code blocks, or emojis. Write your answer as a simple block of unformatted text."
            "Generate the 'company_overview' section for {company_name} strictly based on the following raw sections. "
            "Your output MUST be between 120 and 140 words. Do NOT copy sentences verbatim—synthesize, clarify, and avoid repetition.\n\n"
            "Sections:\n{sections}\n"
        ),
    },
    "key_financials.txt": {
        "sources": ["analyze_income_stmt", "analyze_balance_sheet", "analyze_cash_flow"],
        "word_limit": 155,
        "prompt_template": (
            "IMPORTANT: Reply using only plain text. Do not use any markdown formatting, bold, italics, underlines, bullets, numbers, code blocks, or emojis. Write your answer as a simple block of unformatted text."
            "Generate the 'key_financials' section for {company_name} strictly using the sections below. "
            "Your output MUST be between 120 and 140 words. Do NOT copy sentences verbatim—synthesize and summarize.\n\n"
            "Sections:\n{sections}\n"
        ),
    },
    "valuation.txt": {
        "sources": ["analyze_income_stmt", "analyze_balance_sheet"],
        "word_limit": 155,
        "prompt_template": (
            "IMPORTANT: Reply using only plain text. Do not use any markdown formatting, bold, italics, underlines, bullets, numbers, code blocks, or emojis. Write your answer as a simple block of unformatted text."
            "Create the 'valuation' section for {company_name} using only the provided income statement and balance sheet summaries. "
            "Synthesize the information, avoid repetition, and stay within 120 and 140 words.\n\n"
            "Sections:\n{sections}\n"
        ),
    },
    "risk_assessment.txt": {
        "sources": ["get_risk_assessment"],
        "word_limit": 300,
        "prompt_template": (
            "IMPORTANT: Reply using only plain text. Do not use any markdown formatting, bold, italics, underlines, bullets, numbers, code blocks, or emojis. Write your answer as a simple block of unformatted text."
            "Draft a 'risk_assessment' section for {company_name} using only the following risk content. "
            "The section MUST be between 250 and 350 words. Synthesize and clarify—avoid repeating content and do not copy sentences verbatim.\n\n"
            "Sections:\n{sections}\n"
        ),
    },
    "sell_side_summary.txt": {
        "sources": [
            "analyze_company_description", "analyze_business_highlights", "analyze_income_stmt",
            "analyze_balance_sheet", "analyze_cash_flow", "get_risk_assessment", "get_competitors_analysis"
        ],
        "word_limit": 300,
        "prompt_template": (
            "IMPORTANT: Reply using only plain text. Do not use any markdown formatting, bold, italics, underlines, bullets, numbers, code blocks, or emojis. Write your answer as a simple block of unformatted text."
            "Write a 'sell_side_summary' for {company_name} based on ALL the provided sections below. "
            "The section MUST be between 250 and 350 words. Synthesize all relevant facts into a compelling narrative, avoid verbatim copying or redundancy.\n\n"
            "Sections:\n{sections}\n"
        ),
    },
    "competitors_analysis.txt": {
        "sources": ["get_competitors_analysis"],
        "word_limit": 300,
        "prompt_template": (
            "IMPORTANT: Reply using only plain text. Do not use any markdown formatting, bold, italics, underlines, bullets, numbers, code blocks, or emojis. Write your answer as a simple block of unformatted text."
            "Generate a 'competitors_analysis' section for {company_name} using only the provided competitor analysis. "
            "The section MUST be between 250 and 350 words. Synthesize, clarify, and avoid repetition.\n\n"
            "Sections:\n{sections}\n"
        ),
    }
}
