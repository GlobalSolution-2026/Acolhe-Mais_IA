
"""API de IA do Acolhe+: previsão de falta e verificação de urgência."""
import pandas as pd
from pathlib import Path
import joblib
from flask import Flask, request, jsonify

def validar_campos(dados, obrigatorios):
    if not isinstance(dados, dict):
        raise ValueError('Envie um objeto JSON.')
    ausentes = set(obrigatorios) - set(dados)
    if ausentes: raise ValueError('Campos ausentes: ' + ', '.join(sorted(ausentes)))

def numero(dados, nome, minimo, maximo, inteiro=False):
    valor = dados.get(nome)
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ValueError(f'{nome} deve ser numérico.')
    if not minimo <= valor <= maximo:
        raise ValueError(f'{nome} deve estar entre {minimo} e {maximo}.')
    if inteiro and valor != int(valor): raise ValueError(f'{nome} deve ser inteiro.')
    return int(valor) if inteiro else float(valor)

def binario(dados, nome):
    valor = dados.get(nome)
    if type(valor) is bool: return int(valor)
    if type(valor) is int and valor in (0, 1): return valor
    raise ValueError(f'{nome} deve ser true/false ou 0/1.')

def sexo(dados):
    valor = dados.get('sexo')
    if not isinstance(valor, str) or valor.strip().upper() not in ['F', 'M']:
        raise ValueError('sexo deve ser F ou M, categorias cobertas pela base.')
    return valor.strip().upper()

def pontuar(pacote, valores):
    entrada = pd.DataFrame([valores], columns=pacote['atributos'])
    indice = list(pacote['modelo'].classes_).index(1)
    p = float(pacote['modelo'].predict_proba(entrada)[0, indice])
    return p

def verificar_urgencia(dados, pacote):
    validar_campos(dados, ['idade', 'sexo', 'trauma', 'dor', 'alerta'])
    valores = {
        'idade': numero(dados, 'idade', 18, 120, inteiro=True),
        'sexo': sexo(dados), 'trauma': binario(dados, 'trauma'), 'dor': binario(dados, 'dor'),
        'alerta': binario(dados, 'alerta')
    }
    if valores['dor'] == 1:
        valores['nota_dor'] = numero(dados, 'nota_dor', 1, 10, inteiro=True)
    else:
        valores['nota_dor'] = 0
    p = pontuar(pacote, valores)
    return {'urgente': bool(p >= pacote['limiar']),
            'pontuacao_risco': round(p, 4), 'limiar': float(pacote['limiar'])}


def faixa_da_antecedencia(dias):
    if dias == 0: return 'mesmo_dia'
    if dias <= 7: return '1_a_7'
    if dias <= 30: return '8_a_30'
    return '31_ou_mais'

def prever_falta(dados, pacote):
    validar_campos(dados, ['idade', 'sexo', 'antecedencia_dias', 'dia_semana',
                           'sms_recebido', 'faltas_anteriores'])
    dia = dados.get('dia_semana')
    if isinstance(dia, str) and dia.strip() in [str(i) for i in range(5)]:
        dia = int(dia.strip())
    else:
        dia = numero(dados, 'dia_semana', 0, 4, inteiro=True)
    bairro = dados.get('bairro', 'OUTROS')
    if not isinstance(bairro, str):
        raise ValueError('bairro deve ser texto.')
    bairro = bairro.strip().upper()
    bairro = bairro if bairro in pacote['bairros_conhecidos'] else 'OUTROS'
    antecedencia = numero(dados, 'antecedencia_dias', 0, 3650, inteiro=True)
    valores = {
        'idade': numero(dados, 'idade', 18, 120, inteiro=True), 'sexo': sexo(dados),
        'antecedencia_dias': antecedencia,
        'faixa_antecedencia': faixa_da_antecedencia(antecedencia),
        'dia_semana': str(dia), 'bairro': bairro,
        'sms_recebido': binario(dados, 'sms_recebido'),
        'faltas_anteriores': numero(dados, 'faltas_anteriores', 0, 100000, inteiro=True)
    }
    p = pontuar(pacote, valores)
    risco = 'alto' if p >= pacote['limiar'] else 'baixo'
    return {'pontuacao_risco': round(p, 4), 'risco': risco,
            'limiar': float(pacote['limiar']),
            'confirmacao_dias_uteis': int(pacote['confirmacao_dias_uteis'][risco])}


def criar_app(pasta_modelos=None):
    pasta = Path(pasta_modelos) if pasta_modelos else Path(__file__).resolve().parents[1] / 'modelos'
    pacotes = {}
    for tarefa in ['falta', 'urgencia']:
        arquivo = pasta / f'modelo_{tarefa}.joblib'
        pacote = joblib.load(arquivo)
        pacotes[tarefa] = pacote

    app = Flask(__name__)
    app.json.ensure_ascii = False

    @app.get('/status')
    def status():
        return jsonify(status='ok', modelos={t:p['nome_modelo'] for t,p in pacotes.items()},
                       saida='pontuacao_de_risco_entre_0_e_1')

    @app.post('/prever-falta')
    def falta():
        try:
            return jsonify(prever_falta(request.get_json(silent=True), pacotes['falta']))
        except ValueError as erro:
            return jsonify(erro=str(erro)), 400

    @app.post('/verificar-urgencia')
    def urgencia():
        try:
            return jsonify(verificar_urgencia(request.get_json(silent=True), pacotes['urgencia']))
        except ValueError as erro:
            return jsonify(erro=str(erro)), 400

    return app

if __name__ == '__main__':
    criar_app().run(host='127.0.0.1', port=5000, debug=False)
