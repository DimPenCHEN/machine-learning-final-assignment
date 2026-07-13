import numpy as np
import pandas as pd


def save_svg(path, dates, truth, pred, title: str) -> None:
    width, height = 1100, 460
    pad_l, pad_r, pad_t, pad_b = 95, 30, 58, 55
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n = len(truth)
    ymin = float(min(truth.min(), pred.min()))
    ymax = float(max(truth.max(), pred.max()))
    if ymax <= ymin:
        ymax = ymin + 1.0

    def point(i, v):
        x = pad_l + i * (width - pad_l - pad_r) / max(1, n - 1)
        y = pad_t + (ymax - v) * (height - pad_t - pad_b) / (ymax - ymin)
        return f"{x:.2f},{y:.2f}"

    truth_pts = " ".join(point(i, v) for i, v in enumerate(truth))
    pred_pts = " ".join(point(i, v) for i, v in enumerate(pred))
    y_ticks = []
    for k in range(5):
        val = ymin + (ymax - ymin) * k / 4
        y = pad_t + (ymax - val) * (height - pad_t - pad_b) / (ymax - ymin)
        y_ticks.append((val, y))

    first_date = pd.to_datetime(dates[0]).strftime("%Y-%m-%d")
    last_date = pd.to_datetime(dates[-1]).strftime("%Y-%m-%d")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{pad_l}" y="28" font-family="Arial" font-size="14" fill="#222">{title}</text>',
        f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#333"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#333"/>',
    ]
    for val, y in y_ticks:
        svg.append(f'<line x1="{pad_l}" y1="{y:.2f}" x2="{width-pad_r}" y2="{y:.2f}" stroke="#e6e6e6"/>')
        svg.append(
            f'<text x="{pad_l-12}" y="{y+4:.2f}" text-anchor="end" '
            f'font-family="Arial" font-size="12" fill="#555">{val:.1f}</text>'
        )
    svg.extend(
        [
            f'<polyline fill="none" stroke="#111827" stroke-width="2" points="{truth_pts}"/>',
            f'<polyline fill="none" stroke="#dc2626" stroke-width="2" points="{pred_pts}"/>',
            f'<text x="{pad_l}" y="{height-18}" font-family="Arial" font-size="13" fill="#555">{first_date}</text>',
            f'<text x="{width-145}" y="{height-18}" font-family="Arial" font-size="13" fill="#555">{last_date}</text>',
            f'<rect x="{width-330}" y="12" width="300" height="28" fill="white" stroke="#ddd"/>',
            f'<line x1="{width-314}" y1="27" x2="{width-280}" y2="27" stroke="#111827" stroke-width="2"/>',
            f'<text x="{width-272}" y="31" font-family="Arial" font-size="10" fill="#333">Ground Truth</text>',
            f'<line x1="{width-150}" y1="27" x2="{width-116}" y2="27" stroke="#dc2626" stroke-width="2"/>',
            f'<text x="{width-108}" y="31" font-family="Arial" font-size="10" fill="#333">Prediction</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(svg), encoding="utf-8")
