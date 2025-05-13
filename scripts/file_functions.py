import xml.etree.ElementTree as ET
import os

def function_on_file(path):
    with open(path, "r") as archivo:
        linhas = archivo.readlines()
    linhas.sort()
    return linhas

def convert_cvat_xml_to_abs(xml_path, output_dir, decimals: int = 2):
    """
    Convert a CVAT 1.1 interpolation-format XML file into plain–text files
    containing *absolute* bounding-box coordinates (xtl  ytl  xbr  ybr).

    • One output .txt per frame:  frame_000000.txt, frame_000001.txt, …
    • Lines inside each file preserve the order they appear in the XML.
    • No class-ids, no normalisation.

    Parameters
    ----------
    xml_path   : str  – path to the CVAT XML annotation.
    output_dir : str  – folder where the frame_XXXXXX.txt files will be saved.
    decimals   : int  – number of decimal places to keep (default 2, matching
                         your example).
    """
    # 1️⃣ Parse the XML once
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 2️⃣ Collect all lines per frame
    frame_dict = {}  # {frame_idx: [ "xtl ytl xbr ybr", … ]}

    for track in root.findall("track"):
        for box in track.findall("box"):
            # Skip boxes that are tagged “outside=1”
            if int(box.get("outside", 0)):
                continue

            frame_idx = int(box.get("frame"))
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            fmt = f"{{:.{decimals}f}} {{:.{decimals}f}} {{:.{decimals}f}} {{:.{decimals}f}}"
            line = fmt.format(xtl, ytl, xbr, ybr)

            frame_dict.setdefault(frame_idx, []).append(line)

    # 3️⃣ Write out one text file per frame
    os.makedirs(output_dir, exist_ok=True)
    for frame_idx, lines in frame_dict.items():
        out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))

    print(f"Conversion complete! Absolute-coordinate annotations saved to: {output_dir}")

