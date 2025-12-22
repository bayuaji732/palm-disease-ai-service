def build_explanation_prompt(label: str, confidence: float) -> str:
    if label.lower() == "healthy":
        return """
        Jelaskan tanda-tanda bahwa tanaman sawit dalam kondisi sehat. 
        Berikan informasi singkat mengenai ciri-ciri fisik dan 
        pertumbuhan yang menunjukkan kesehatan optimal tanaman sawit.
        """
    
    else:
        return """
        Langsung Berikan informasi singkat mengenai penyakit tanaman sawit: {disease}. 
        Tuliskan dalam format:
        1. Deskripsi, 2. Efek, 3. Rekomendasi. 
        Gunakan bahasa Indonesia formal.
        """.format(disease=label)