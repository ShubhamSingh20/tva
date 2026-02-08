import xml.etree.ElementTree as ET
from typing import Dict, Any, List
import json
from .tui import console

def xml_element_to_dict(element: ET.Element) -> Dict[str, Any]:
    """Recursively convert an XML ElementTree element to a dictionary."""
    result: Dict[str, Any] = {}

    # Add attributes prefixed with '@'
    if element.attrib:
        for key, value in element.attrib.items():
            result[f"@{key}"] = value

    # Add text content
    text = (element.text or "").strip()
    children = list(element)

    if not children:
        # Leaf node: return text if no attributes, else add '#text'
        if text:
            if result:
                result["#text"] = text
            else:
                return text
        return result or None

    # Group children by tag
    child_map: Dict[str, List[Any]] = {}
    for child in children:
        child_dict = xml_element_to_dict(child)
        child_map.setdefault(child.tag, []).append(child_dict)

    # Flatten single-item lists, keep multi-item as lists
    for tag, items in child_map.items():
        result[tag] = items if len(items) > 1 else items[0]

    if text:
        result["#text"] = text

    return result


def load_xml_as_dict(xml_path: str) -> Dict[str, Any]:
    """Parse an XML file and return it as a nested dictionary."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return {root.tag: xml_element_to_dict(root)}


def load_json(json_path: str) -> Dict[str, Any]:
    """Load a JSON file and return it as a dictionary."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_as_json(content: str) -> Dict[str, Any]:
    """Load a JSON string and return it as a dictionary."""
    content = content.strip('```json\n').strip('\n```').strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        console.print(f"[bold red]Failed to parse JSON:[/] {content}")
        return None
