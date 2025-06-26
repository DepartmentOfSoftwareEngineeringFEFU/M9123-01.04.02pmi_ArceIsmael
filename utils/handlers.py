import re
import json
from typing import List, Any, Dict, Optional


def extract_text_between_quotes(text: str) -> str:
    quoted_matches = re.findall('"([^"]*)"', text)
    if quoted_matches:
        return quoted_matches[0]

    cleaned_text = text.strip()

    prefixes_to_remove = [
        "The section heading that contains",
        "The heading is:",
        "The treatment section is:",
        "Section:",
        "Heading:",
    ]

    for prefix in prefixes_to_remove:
        if cleaned_text.lower().startswith(prefix.lower()):
            cleaned_text = cleaned_text[len(prefix) :].strip()
            break

    return cleaned_text


def extract_section_markers(llm_response):
    if "No next section found" in llm_response or "I cannot answer" in llm_response:
        return None, None

    pattern = r'[\'"]([^\'"]+)[\'"]'  # Matches text between single (or double) quotes
    matches = re.findall(pattern, llm_response)

    if len(matches) >= 2:
        start_marker = matches[0]
        end_marker = matches[1]
        return start_marker, end_marker

    section_pattern = r"(\d+\.\s+[А-Яа-я][^\.]*(?:\.|$))"
    section_matches = re.findall(section_pattern, llm_response)

    if len(section_matches) >= 2:
        start_marker = section_matches[0].strip().rstrip(".")
        end_marker = section_matches[1].strip().rstrip(".")
        return start_marker, end_marker

    lines = llm_response.strip().split("\n")
    potential_sections = []

    for line in lines:
        line = line.strip()
        if re.match(r"^\d+\.\s+[А-Яа-я]", line):
            potential_sections.append(line)

    if len(potential_sections) >= 2:
        return potential_sections[0], potential_sections[1]

    return None, None


def remove_trailing_dots(text):
    return re.sub(r"\.+$", "", text)


def extract_section_content(main_text, start_marker, end_marker):
    """
    Extract section content with flexible matching to handle:
    - Line breaks within section titles
    - Skipping table of contents to find actual content
    - Stopping at the correct section boundary
    """
    # First, try to find all occurrences using a simplified marker
    # Extract the basic section pattern like "3. Лечение" from the full marker
    simplified_start = extract_simple_marker(start_marker)

    # Find all occurrences of the simplified marker in the original text
    start_positions = find_all_positions(main_text, simplified_start)

    if not start_positions:
        normalized_main_text = normalize_for_matching(main_text)
        normalized_start = normalize_for_matching(start_marker)
        start_positions = find_all_positions(normalized_main_text, normalized_start)
        start_positions = [
            find_original_position(main_text, normalized_main_text, pos) for pos in start_positions
        ]
        start_positions = [pos for pos in start_positions if pos != -1]

    print(
        f"🔍 DEBUG: Found {len(start_positions)} start marker occurrences at positions: {start_positions}"
    )

    # Analyze each start position to determine if it's TOC or actual content
    candidates = []

    for i, original_start_pos in enumerate(start_positions):
        if original_start_pos == -1:
            continue

        preview = main_text[max(0, original_start_pos - 500) : original_start_pos + 1500]

        lines = preview.split("\n")
        numbered_lines = 0
        for line in lines:
            line = line.strip()
            if re.match(r"^\d+\.\s+[А-Яа-я]", line) or re.match(r"^\d+\.\d+\s+[А-Яа-я]", line):
                numbered_lines += 1

        has_treatment_content = any(
            keyword in preview.lower()
            for keyword in [
                "рекомендуется",
                "не рекомендуется",
                "уровень убедительности",
                "препарат",
                "доза",
                "терапия",
                "пациентам",
            ]
        )

        toc_score = numbered_lines  # Higher = more likely TOC
        content_score = 0
        if has_treatment_content:
            content_score += 5
        if original_start_pos > len(main_text) * 0.3:  # Later in document
            content_score += 3

        candidates.append(
            {
                "position": original_start_pos,
                "index": i,
                "toc_score": toc_score,
                "content_score": content_score,
                "is_likely_toc": toc_score >= 5,
                "is_likely_content": content_score >= 5 and not (toc_score >= 5),
            }
        )

        print(
            f"  Candidate {i+1} at pos {original_start_pos}: TOC_score={toc_score}, Content_score={content_score}, Likely={'TOC' if toc_score >= 5 else 'CONTENT' if content_score >= 5 else 'UNKNOWN'}"
        )

    # Sort candidates: prefer content over TOC, but prioritize EARLIER positions for main sections
    # We want the first valid content section, not the last one
    candidates.sort(
        key=lambda x: (x["is_likely_content"], -x["position"])
    )  # Negative position = earlier first

    for candidate in reversed(candidates):
        original_start_pos = candidate["position"]

        all_candidates_are_toc = all(c["is_likely_toc"] for c in candidates)
        if (
            candidate["is_likely_toc"]
            and any(c["is_likely_content"] for c in candidates)
            and not all_candidates_are_toc
        ):
            print(f"  Skipping TOC candidate at position {original_start_pos}")
            continue

        print(f"  Trying to extract from position {original_start_pos}")

        remaining_text = main_text[original_start_pos:]

        normalized_end = normalize_for_matching(end_marker)
        end_positions = find_all_positions(normalize_for_matching(remaining_text), normalized_end)

        extracted_content = None

        if end_positions:
            # Found end marker, extract content between start and end
            for end_pos in end_positions:
                original_end_pos = find_original_position(
                    remaining_text, normalize_for_matching(remaining_text), end_pos
                )
                if original_end_pos > 0:
                    content = remaining_text[:original_end_pos].strip()
                    if len(content) > 500 and not is_table_of_contents(content):
                        extracted_content = content
                        print(f"    Found content using end marker: {len(content)} chars")
                        break

        if not extracted_content:
            section_number = extract_section_number(start_marker)

            if section_number:
                next_section_num = int(section_number) + 1

                next_section_patterns = [
                    rf"\n\s*{next_section_num}\.\s+[А-Яа-я]",  # Specific next section
                    r"\n\s*4\.\s+[А-Яа-я]",  # Section 4
                    r"\n\s*5\.\s+[А-Яа-я]",  # Section 5
                    r"\n\s*Критерии оценки",  # Quality criteria
                    r"\n\s*Список литературы",  # Bibliography
                    r"\n\s*Приложение",  # Appendix
                ]

                for pattern in next_section_patterns:
                    next_section_match = re.search(pattern, remaining_text)
                    if next_section_match:
                        content = remaining_text[: next_section_match.start()].strip()
                        if len(content) > 500:
                            extracted_content = content
                            print(f"    Found content using section boundary: {len(content)} chars")
                            break

                if not extracted_content:
                    major_break_patterns = [
                        r"\n\s*\d+\.\s+[А-Я][а-я]+\s+[а-я]+",  # Any numbered section with Russian title
                        r"\n\s*[А-Я][а-я]+\s+[а-я]+\s+[а-я]+",  # Capitalized multi-word titles
                    ]

                    for pattern in major_break_patterns:
                        matches = list(re.finditer(pattern, remaining_text))
                        if matches:
                            for match in matches:
                                if match.start() > 1000:
                                    content = remaining_text[: match.start()].strip()
                                    extracted_content = content
                                    print(
                                        f"    Found content using major break: {len(content)} chars"
                                    )
                                    break
                            if extracted_content:
                                break

        if extracted_content:
            if len(extracted_content) < 1000:
                print(
                    f"⚠️  Content is very short ({len(extracted_content)} chars), trying larger extraction..."
                )

                remaining_text_large = main_text[original_start_pos:]

                major_boundaries = [
                    r"\n\s*(?:Критерии оценки|Список литературы|Приложение|Заключение)",
                    r"\n\s*(?:8|9|10)\.\s+[А-Я][а-я]+",  # Much later sections
                ]

                for pattern in major_boundaries:
                    match = re.search(pattern, remaining_text_large[2000:])  # Skip first 2000 chars
                    if match:
                        large_content = remaining_text_large[: 2000 + match.start()].strip()
                        if len(large_content) > 2000:
                            print(
                                f"  ✅ Found larger content section: {len(large_content)} characters"
                            )
                            return large_content
                        break

                if len(remaining_text_large) > 5000:
                    large_content = remaining_text_large[:5000].strip()
                    print(f"  ✅ Extracting large chunk: {len(large_content)} characters")
                    return large_content

            if not is_table_of_contents(extracted_content):
                print(f"✅ Successfully extracted {len(extracted_content)} characters")
                return extracted_content
            else:
                print(f"❌ Extracted content looks like TOC, continuing...")

    print("❌ Could not find suitable content in any candidate")
    return None


def find_all_positions(text, pattern):
    """Find all positions of pattern in text"""
    positions = []
    start = 0
    while True:
        pos = text.find(pattern, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions


def score_content_quality(content, start_marker, position_in_doc=0, total_doc_length=100000):
    """Score content quality to distinguish actual content from table of contents"""
    score = 0

    position_ratio = position_in_doc / total_doc_length
    if position_ratio > 0.3:
        score += 3
    elif position_ratio > 0.1:
        score += 1

    if len(content) > 1000:
        score += 2
    elif len(content) > 500:
        score += 1

    treatment_keywords = [
        "лечение",
        "терапия",
        "препарат",
        "доза",
        "применение",
        "показания",
        "противопоказания",
        "эффективность",
        "безопасность",
        "рекомендуется",
        "пациентам",
        "лекарственных",
        "медицинские",
    ]

    content_lower = content.lower()
    keyword_count = sum(1 for keyword in treatment_keywords if keyword in content_lower)
    score += min(keyword_count, 4)

    if is_table_of_contents(content):
        score -= 10  # Heavy penalty

    lines = content.split("\n")[:10]
    numbered_section_count = 0
    for line in lines:
        line = line.strip()
        if re.match(r"^\d+\.\s+[А-Яа-я]", line):
            numbered_section_count += 1

    if numbered_section_count >= 3:  # If 3+ sections in first 10 lines = likely TOC
        score -= 8

    paragraph_lines = 0
    for line in lines:
        line = line.strip()
        if line and not re.match(r"^\d+\.", line) and len(line) > 50:
            paragraph_lines += 1

    if paragraph_lines > 3:
        score += 3
    elif paragraph_lines > 1:
        score += 1

    if "не рекомендуется" in content_lower or "рекомендуется" in content_lower:
        score += 2

    if "уровень убедительности" in content_lower:
        score += 2

    return score


def extract_section_number(marker):
    """Extract section number from marker, e.g., '3' from '3. Лечение...'"""
    match = re.match(r"^(\d+)\.", marker)
    return match.group(1) if match else None


def is_table_of_contents(text):
    """Check if text looks like a table of contents"""
    lines = text.split("\n")
    numbered_lines = 0
    total_lines = 0
    section_pattern_lines = 0
    subsection_pattern_lines = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        total_lines += 1

        # Count lines that look like table of contents entries
        if re.match(r"^\d+\.\d+\s+[А-Яа-я]", line):  # "2.1 Something"
            numbered_lines += 1
            subsection_pattern_lines += 1
        elif re.match(r"^\d+\.\s+[А-Яа-я]", line):  # "3. Something"
            numbered_lines += 1
            section_pattern_lines += 1
        elif re.match(r"^Приложение\s+[А-Я]", line):  # "Приложение А1"
            numbered_lines += 1
        elif line in ["Список литературы", "Критерии оценки качества медицинской помощи"]:
            numbered_lines += 1

    # Strong indicators of table of contents:
    if total_lines > 3:
        ratio = numbered_lines / total_lines

        # Very high ratio of numbered lines = definitely TOC
        if ratio > 0.7:
            return True

        # Multiple sections and subsections = likely TOC ONLY if high ratio
        # Real content can have subsections too (3.1, 3.2, etc.)
        if section_pattern_lines >= 2 and subsection_pattern_lines >= 2 and ratio > 0.4:
            return True

        # Moderate ratio but with appendix references = likely TOC
        if ratio > 0.5 and ("Приложение" in text or "Список литературы" in text):
            return True
    return False


def extract_simple_marker(full_marker):
    """Extract simple version of marker, e.g., '3. Лечение' from full title"""
    match = re.match(r"^(\d+\.\s+[А-Яа-я]+)", full_marker)
    if match:
        return match.group(1)
    return full_marker[:20]


def find_original_position(original_text, normalized_text, normalized_pos):
    """
    Find the approximate position in original text corresponding to
    a position in normalized text
    """
    if normalized_pos == 0:
        return 0

    chars_counted = 0
    original_pos = 0

    for i, char in enumerate(original_text):
        if chars_counted >= normalized_pos:
            return i

        if char.strip():
            chars_counted += 1
        elif chars_counted > 0 and original_text[i - 1 : i].strip():
            chars_counted += 1

        original_pos = i

    return original_pos


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


def clean_subsections(subsections: List[str], main_section_number: str = None) -> List[str]:
    """
    Clean up extracted subsections by removing duplicates and filtering out foreign text.

    Args:
        subsections: List of extracted subsection strings
        main_section_number: Main section number (e.g., "5" for section 5) to validate patterns

    Returns:
        Cleaned list of subsections
    """
    if not subsections:
        return []

    if not main_section_number and subsections:
        first_match = re.match(r"^(\d+)\.", subsections[0].strip())
        if first_match:
            main_section_number = first_match.group(1)

    cleaned = []
    seen = set()

    for subsection in subsections:
        if not subsection or not subsection.strip():
            continue

        subsection = subsection.strip()

        if subsection in seen:
            continue

        valid_patterns = (
            [
                rf"^{main_section_number}\.\d+\.?\s+[А-Яа-я]",  # 5.1. TITLE or 5.1 TITLE
            ]
            if main_section_number
            else [
                r"^\d+\.\d+\.?\s+[А-Яа-я]",  # X.Y. TITLE or X.Y TITLE
            ]
        )

        nested_patterns = (
            [
                rf"^{main_section_number}\.\d+\.\d+\.?\s+[А-Яа-я]",  # 5.1.1. TITLE or 5.1.1 TITLE
            ]
            if main_section_number
            else [
                r"^\d+\.\d+\.\d+\.?\s+[А-Яа-я]",  # X.Y.Z. TITLE or X.Y.Z TITLE
            ]
        )

        is_nested = any(re.match(pattern, subsection) for pattern in nested_patterns)
        if is_nested:
            continue

        is_valid = any(re.match(pattern, subsection) for pattern in valid_patterns)

        if is_valid:
            if len(subsection) > 500:
                continue

            recommendation_indicators = [
                "рекомендуется",
                "следует",
                "необходимо проводить",
                "пациентам",
                "при назначении",
                "характерно более",
                "абсолютно обосновано",
                "нуждаются в",
            ]

            if len(subsection) > 200 and any(
                indicator in subsection.lower() for indicator in recommendation_indicators
            ):
                continue

            if not re.search(r"[А-Я].*[А-Я]", subsection) and len(subsection) > 100:
                continue

        if is_valid:
            cleaned.append(subsection)
            seen.add(subsection)

    return cleaned


def safe_json_parse(response: str, context: str = "response") -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON response with fallback handling for common issues.

    Args:
        response: The JSON string to parse
        context: Context description for logging (e.g., "LLM response", "API response")

    Returns:
        Parsed JSON object or None if parsing fails
    """
    if not response or not response.strip():
        print(f"⚠️  Empty {context} provided for JSON parsing")
        return None

    try:
        clean_response = response.strip()

        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        elif clean_response.startswith("```"):
            clean_response = clean_response[3:]

        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]

        return json.loads(clean_response.strip())

    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parsing failed for {context}: {str(e)}")
        print(f"📤 Raw {context}: {response[:300]}...")

        try:
            fixed_response = _fix_common_json_issues(clean_response)
            return json.loads(fixed_response)
        except json.JSONDecodeError:
            print(f"❌ Failed to fix JSON issues in {context}")
            return None


def _fix_common_json_issues(json_str: str) -> str:
    """
    Attempt to fix common JSON formatting issues.

    Args:
        json_str: The malformed JSON string

    Returns:
        Potentially fixed JSON string
    """
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    if json_str.count('"') % 2 != 0:
        json_str += '"'

    # Ensure proper bracket closure
    open_braces = json_str.count("{")
    close_braces = json_str.count("}")
    if open_braces > close_braces:
        json_str += "}" * (open_braces - close_braces)

    open_brackets = json_str.count("[")
    close_brackets = json_str.count("]")
    if open_brackets > close_brackets:
        json_str += "]" * (open_brackets - close_brackets)

    return json_str
