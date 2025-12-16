def build_explanation_prompt(label: str, confidence: float) -> str:
    return (
        f"Kondisi terdeteksi: {label}\n"
        f"Tingkat keyakinan: {confidence*100:.0f}%\n"
        "Jelaskan kondisi ini secara singkat."
    )