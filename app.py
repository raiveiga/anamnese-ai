from flask import Flask, request, jsonify
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

@app.route("/transcrever", methods=["POST"])
def transcrever_audio():
    audio = request.files["audio"]
    transcript = openai.Audio.transcribe("whisper-1", audio)
    texto_transcrito = transcript["text"]
    return jsonify({"transcricao": texto_transcrito})

@app.route("/gerar_anamnese", methods=["POST"])
def gerar_anamnese():
    dados = request.json
    texto_transcricao = dados["texto"]

    prompt = f"""
    A seguir, está a transcrição de uma conversa entre médico e paciente. 
    Gere uma ficha de anamnese pediátrica completa com os seguintes campos:
    - Nome do paciente
    - Idade
    - Queixa principal
    - HDA
    - Antecedentes pessoais e familiares
    - Alergias
    - Hábitos de vida
    - Revisão de sistemas
    - Hipóteses diagnósticas (vazias)
    - Conduta (vazia)
    
    Transcrição: {texto_transcricao}
    """

    resposta = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    ficha = resposta.choices[0].message.content
    return jsonify({"anamnese": ficha})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
