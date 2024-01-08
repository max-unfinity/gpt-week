import re

def remove_numeration(summary):
    summary_cleaned = re.sub(r'^\d+\.\s+', '', summary, flags=re.MULTILINE)
    return summary_cleaned

def remove_prompt(summary):
    summary_cleaned = summary.replace('Основные мысли статьи:\n\n', '').strip()
    summary_cleaned = summary_cleaned.replace('Основные мысли из статьи:', '').strip()
    return summary_cleaned

def substitute_newlines(summary, char='- '):
    summary_cleaned = summary.replace('\n', char).strip()
    return summary_cleaned

def remove_ticks(summary):
    summary_cleaned = summary.replace('-', '').strip()
    return summary_cleaned


def post_process_yagpt_lite(summary, add_dash=False):
    summary_cleaned = summary
    summary_cleaned = remove_numeration(summary_cleaned)  # +1%
    summary_cleaned = remove_prompt(summary_cleaned)  # +0.7%
    char = '- ' if add_dash else ' '
    summary_cleaned = substitute_newlines(summary_cleaned, char=char)  # +1% if add_dash
    # summary_cleaned = remove_ticks(summary_cleaned)  # -1%
    # summary_cleaned = "- "+summary_cleaned  # -0.5%
    return summary_cleaned

def post_process_yagpt_summarization(summary):
    summary_cleaned = summary
    summary_cleaned = re.sub(r'^-\s+', '', summary_cleaned, flags=re.MULTILINE)
    summary_cleaned = summary_cleaned.replace('\n', ' ').strip()
    return summary_cleaned