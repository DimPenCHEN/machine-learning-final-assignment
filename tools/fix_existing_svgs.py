import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"

OLD_X0, OLD_X1 = 70.0, 1075.0
NEW_X0, NEW_X1 = 95.0, 1070.0
NEW_TOP, NEW_BOTTOM = 52.0, 405.0


def transform_points(points: str) -> str:
    parsed = []
    for item in points.split():
        x_str, y_str = item.split(",")
        parsed.append((float(x_str), float(y_str)))
    xs = [x for x, _ in parsed]
    xmin, xmax = min(xs), max(xs)
    span = max(1.0, xmax - xmin)
    pairs = []
    for x, y in parsed:
        mapped_x = NEW_X0 + (x - xmin) * (NEW_X1 - NEW_X0) / span
        pairs.append(f"{mapped_x:.2f},{y:.2f}")
    return " ".join(pairs)


def fix_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r'<text x="(?:70|95)" y="28" font-family="Arial" font-size="(?:20|18|14)" fill="#222">([^<]+)</text>',
        r'<text x="95" y="28" font-family="Arial" font-size="14" fill="#222">\1</text>',
        text,
    )
    text = re.sub(r'<line x1="70" y1="405" x2="1075" y2="405" stroke="#333"/>',
                  '<line x1="95" y1="405" x2="1070" y2="405" stroke="#333"/>', text)
    text = re.sub(r'<line x1="70" y1="45" x2="70" y2="405" stroke="#333"/>',
                  '<line x1="95" y1="58" x2="95" y2="405" stroke="#333"/>', text)
    text = re.sub(r'<line x1="95" y1="52" x2="95" y2="405" stroke="#333"/>',
                  '<line x1="95" y1="58" x2="95" y2="405" stroke="#333"/>', text)
    text = re.sub(r'<line x1="70" y1="([^"]+)" x2="1075" y2="\1" stroke="#e6e6e6"/>',
                  r'<line x1="95" y1="\1" x2="1070" y2="\1" stroke="#e6e6e6"/>', text)
    text = re.sub(
        r'<text x="8" y="([^"]+)" font-family="Arial" font-size="12" fill="#555">([^<]+)</text>',
        r'<text x="83" y="\1" text-anchor="end" font-family="Arial" font-size="12" fill="#555">\2</text>',
        text,
    )
    text = re.sub(
        r'points="([^"]+)"',
        lambda m: f'points="{transform_points(m.group(1))}"',
        text,
    )
    text = re.sub(r'<text x="70" y="442"', '<text x="95" y="442"', text)
    text = re.sub(r'<text x="965" y="442"', '<text x="955" y="442"', text)
    text = re.sub(
        r'<rect x="(?:840|735)" y="(?:14|12)" width="(?:220|335)" height="(?:28|30)" fill="white" stroke="#ddd"/>',
        '<rect x="770" y="12" width="300" height="28" fill="white" stroke="#ddd"/>',
        text,
    )
    text = re.sub(
        r'<line x1="(?:852|752)" y1="28" x2="(?:892|790)" y2="28" stroke="#111827" stroke-width="2"/>',
        '<line x1="786" y1="27" x2="820" y2="27" stroke="#111827" stroke-width="2"/>',
        text,
    )
    text = re.sub(r'<line x1="786" y1="28" x2="820" y2="28" stroke="#111827" stroke-width="2"/>',
                  '<line x1="786" y1="27" x2="820" y2="27" stroke="#111827" stroke-width="2"/>', text)
    text = re.sub(
        r'<text x="(?:898|798)" y="32" font-family="Arial" font-size="(?:13|12)" fill="#333">Ground Truth</text>',
        '<text x="828" y="31" font-family="Arial" font-size="10" fill="#333">Ground Truth</text>',
        text,
    )
    text = re.sub(r'<text x="828" y="32" font-family="Arial" font-size="12" fill="#333">Ground Truth</text>',
                  '<text x="828" y="31" font-family="Arial" font-size="10" fill="#333">Ground Truth</text>', text)
    text = re.sub(
        r'<line x1="(?:995|930)" y1="28" x2="(?:1035|968)" y2="28" stroke="#dc2626" stroke-width="2"/>',
        '<line x1="950" y1="27" x2="984" y2="27" stroke="#dc2626" stroke-width="2"/>',
        text,
    )
    text = re.sub(r'<line x1="950" y1="28" x2="984" y2="28" stroke="#dc2626" stroke-width="2"/>',
                  '<line x1="950" y1="27" x2="984" y2="27" stroke="#dc2626" stroke-width="2"/>', text)
    text = re.sub(
        r'<text x="(?:1041|976)" y="32" font-family="Arial" font-size="(?:13|12)" fill="#333">Prediction</text>',
        '<text x="992" y="31" font-family="Arial" font-size="10" fill="#333">Prediction</text>',
        text,
    )
    text = re.sub(r'<text x="992" y="32" font-family="Arial" font-size="12" fill="#333">Prediction</text>',
                  '<text x="992" y="31" font-family="Arial" font-size="10" fill="#333">Prediction</text>', text)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    for path in sorted(FIG_DIR.glob("*.svg")):
        fix_file(path)
        print(f"fixed {path.name}")


if __name__ == "__main__":
    main()
