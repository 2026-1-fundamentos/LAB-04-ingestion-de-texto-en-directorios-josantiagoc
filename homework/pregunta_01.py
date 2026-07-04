# pylint: disable=import-outside-toplevel
# pylint: disable=line-too-long
# flake8: noqa
"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta.
"""

def pregunta_01():
    """
    La información requerida para este laboratio esta almacenada en el
    archivo "files/input.zip" ubicado en la carpeta raíz.
    Descomprima este archivo.

    Como resultado se creara la carpeta "input" en la raiz del
    repositorio, la cual contiene la siguiente estructura de archivos:


    ```
    train/
        negative/
            0000.txt
            0001.txt
            ...
        positive/
            0000.txt
            0001.txt
            ...
        neutral/
            0000.txt
            0001.txt
            ...
    test/
        negative/
            0000.txt
            0001.txt
            ...
        positive/
            0000.txt
            0001.txt
            ...
        neutral/
            0000.txt
            0001.txt
            ...
    ```

    A partir de esta informacion escriba el código que permita generar
    dos archivos llamados "train_dataset.csv" y "test_dataset.csv". Estos
    archivos deben estar ubicados en la carpeta "output" ubicada en la raiz
    del repositorio.

    Estos archivos deben tener la siguiente estructura:

    * phrase: Texto de la frase. hay una frase por cada archivo de texto.
    * sentiment: Sentimiento de la frase. Puede ser "positive", "negative"
      o "neutral". Este corresponde al nombre del directorio donde se
      encuentra ubicado el archivo.

    Cada archivo tendria una estructura similar a la siguiente:

    ```
    |    | phrase                                                                                                                                                                 | target   |
    |---:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------|
    |  0 | Cardona slowed her vehicle , turned around and returned to the intersection , where she called 911                                                                     | neutral  |
    |  1 | Market data and analytics are derived from primary and secondary research                                                                                              | neutral  |
    |  2 | Exel is headquartered in Mantyharju in Finland                                                                                                                         | neutral  |
    |  3 | Both operating profit and net sales for the three-month period increased , respectively from EUR16 .0 m and EUR139m , as compared to the corresponding quarter in 2006 | positive |
    |  4 | Tampere Science Parks is a Finnish company that owns , leases and builds office properties and it specialises in facilities for technology-oriented businesses         | neutral  |
    ```


    """

    import glob
    import os
    import re
    import zipfile
    import nltk
    import pandas as pd
    from nltk.tokenize import word_tokenize


    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab")

    zip_path = "files/input.zip"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("files")  # Se extrae dentro de la carpeta files/

    def load_data(input_directory):
        """Carga datos de subcarpetas usando rutas profundas (*.txt)"""
        sequence = []

        files = glob.glob(f"{input_directory}/**/*.txt", recursive=True)

        for file in files:
            file_clean = file.replace("\\", "/")
            if os.path.isfile(file_clean):

                sentiment = file_clean.split("/")[-2]
                with open(file_clean, "rt", encoding="utf-8") as f:
                    raw_text = f.read()
                    sequence.append((file_clean, sentiment, raw_text))
        return sequence

    def clean_text(sequence):
        cleaned_sequence = []
        for file, sentiment, text in sequence:
            cleaned_text = re.sub(r"\n", " ", text)
            cleaned_text = re.sub(r"\s+", " ", cleaned_text)
            cleaned_text = cleaned_text.strip()
            cleaned_text = cleaned_text.lower()
            cleaned_sequence.append((file, sentiment, cleaned_text))
        return cleaned_sequence

    def tokenize(sequence):
        tokenized_sequence = []
        for file, sentiment, text in sequence:
            tokens = word_tokenize(text)
            tokenized_sequence.append((file, sentiment, tokens))
        return tokenized_sequence

    def filter_tokens_b(sequence):
        filtered_sequence = []
        for file, sentiment, tokens in sequence:
            filtered_tokens = [
                re.sub(r"[^a-zA-Z\s]", " ", token) for token in tokens
            ]
            filtered_tokens = [
                re.sub(r"\s+", " ", token) for token in filtered_tokens
            ]
            filtered_tokens = [token.strip() for token in filtered_tokens]
            filtered_tokens = [token for token in filtered_tokens if token != ""]
            filtered_sequence.append((file, sentiment, filtered_tokens))
        return filtered_sequence

    def remove_stopwords(sequence):
        stop_words = set(nltk.corpus.stopwords.words("english"))
        filtered_sequence = []
        for file, sentiment, tokens in sequence:
            filtered_tokens = [
                token for token in tokens if token not in stop_words
            ]
            filtered_sequence.append((file, sentiment, filtered_tokens))
        return filtered_sequence

    def run_pipeline(input_dir):
        """Ejecuta secuencialmente la limpieza sobre la secuencia de datos"""
        seq = load_data(input_dir)
        seq = clean_text(seq)
        seq = tokenize(seq)
        seq = filter_tokens_b(seq)
        seq = remove_stopwords(seq)
        return seq

    def save_to_csv(sequence, output_filepath):
        """Guarda la secuencia final en formato CSV con el índice requerido"""
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        data_records = []
        for _, sentiment, tokens in sequence:
            phrase = " ".join(tokens)
            data_records.append({"phrase": phrase, "target": sentiment})

        df = pd.DataFrame(data_records)

        df.to_csv(output_filepath, index=True, index_label="")

    train_seq = run_pipeline("files/input/train")
    save_to_csv(train_seq, "files/output/train_dataset.csv")

    test_seq = run_pipeline("files/input/test")
    save_to_csv(test_seq, "files/output/test_dataset.csv")

pregunta_01()
