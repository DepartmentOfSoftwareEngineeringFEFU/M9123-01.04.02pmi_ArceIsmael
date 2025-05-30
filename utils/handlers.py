import re


def extract_text_between_quotes(text: str) -> str:
    return re.findall('"([^"]*)"', text)[0]


def extract_section_markers(llm_response):
    pattern = r'[\'"]([^\'"]+)[\'"]'  # Matches text between single (or double) quotes
    matches = re.findall(pattern, llm_response)

    if len(matches) >= 2:
        start_marker = matches[0]
        end_marker = matches[1]
        return start_marker, end_marker
    return None, None


def remove_trailing_dots(text):
    return re.sub(r"\.+$", "", text)


def extract_section_content(main_text, start_marker, end_marker):
    start_marker_escaped = re.escape(start_marker)
    end_marker_escaped = re.escape(end_marker)

    pattern = f"({start_marker_escaped}.*?)(?={end_marker_escaped})"
    matches = re.finditer(pattern, main_text, re.DOTALL)

    matches_list = list(matches)

    if not matches_list:
        return None

    # we take the last match cuz the document can contain table of contents
    return matches_list[-1].group(1)


def normalize_for_matching(text):
    """Normalize text for matching by removing extra spaces and newlines."""
    return re.sub(r"\s+", " ", text).strip()


def split_by_subsections(text, subsections):
    normalized_text = normalize_for_matching(text)
    normalized_subsections = [normalize_for_matching(sub) for sub in subsections]

    sections = {}

    positions = []
    for norm_sub, original_sub in zip(normalized_subsections, subsections):
        pos = normalized_text.find(norm_sub)
        if pos != -1:
            positions.append((pos, original_sub))

    positions.sort()

    for i in range(len(positions)):
        current_pos, current_section = positions[i]

        if i < len(positions) - 1:
            next_pos = positions[i + 1][0]
            content = text[
                text.find(current_section) + len(current_section) : text.find(positions[i + 1][1])
            ].strip()
        else:
            content = text[text.find(current_section) + len(current_section) :].strip()

        sections[current_section] = content

    return sections
