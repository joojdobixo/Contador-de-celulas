"""
Contador de Células via Watershed
Projeto de Extensão - Cálculo III (DIN00018) | UNIR/DACC 2026.2

Pipeline: escala de cinza -> blur -> threshold (Otsu) -> limpeza morfológica
          -> distance transform -> marcadores -> watershed -> contagem/área

Cada etapa é explicada nos comentários com o conceito matemático correspondente.
"""

import cv2
import numpy as np


def carregar_e_preparar(caminho_imagem):
    """Carrega a imagem e converte para escala de cinza + suaviza (blur).

    Conceito: a imagem vira uma função escalar f(x, y) de intensidade.
    O blur é uma suavização local (evita que ruído pequeno seja
    confundido com célula/borda real).
    """
    img = cv2.imread(caminho_imagem)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {caminho_imagem}")
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(cinza, (5, 5), 0)
    return img, cinza, suave


def binarizar(img_suave):
    """Separa 'célula' de 'fundo' usando threshold automático (Otsu).

    Otsu escolhe o limiar que melhor separa dois grupos de intensidade
    (fundo e objeto), maximizando a variância entre eles.
    """
    _, binaria = cv2.threshold(
        img_suave, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binaria


def limpar_ruido(binaria, tamanho_kernel=3, iteracoes=2):
    """Remove pequenos ruídos da máscara binária com abertura morfológica
    (erosão seguida de dilatação).
    """
    kernel = np.ones((tamanho_kernel, tamanho_kernel), np.uint8)
    limpa = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=iteracoes)
    return limpa


def gerar_marcadores(limpa, fator_distancia=0.5):
    """Gera os marcadores necessários para o Watershed.

    Distance Transform: cada pixel do objeto recebe um valor = distância
    até o pixel de fundo mais próximo. Isso cria um "campo escalar" cujos
    máximos locais tendem a ficar no CENTRO de cada célula.

    Threshold nesse campo isola esses centros = marcadores de "foreground"
    (um marcador por célula, mesmo que estejam encostadas).
    """
    dist = cv2.distanceTransform(limpa, cv2.DIST_L2, 5)
    _, marcadores_fg = cv2.threshold(
        dist, fator_distancia * dist.max(), 255, 0
    )
    marcadores_fg = np.uint8(marcadores_fg)

    # área de fundo certa (dilatando a máscara original)
    fundo_certo = cv2.dilate(limpa, np.ones((3, 3), np.uint8), iterations=3)

    # região desconhecida = nem certamente fundo, nem certamente célula
    # (é exatamente onde o watershed vai decidir a fronteira entre células)
    fundo_desconhecido = cv2.subtract(fundo_certo, marcadores_fg)

    # rotula cada marcador de célula com um número (componentes conectados)
    _, marcadores = cv2.connectedComponents(marcadores_fg)

    # soma 1 para que o fundo seja 1 em vez de 0
    # (o watershed reserva o valor 0 para a fronteira desconhecida)
    marcadores = marcadores + 1
    marcadores[fundo_desconhecido == 255] = 0

    return marcadores, dist


def aplicar_watershed(img_original, marcadores):
    """Aplica o algoritmo Watershed.

    Interpreta a imagem (via marcadores de distância) como uma superfície
    topográfica: os "vales" são os centros das células e o algoritmo
    "inunda" a partir deles até encontrar a fronteira com a célula vizinha,
    que fica marcada com o valor -1 (linha de watershed).
    """
    marcadores_resultado = cv2.watershed(img_original, marcadores.copy())
    return marcadores_resultado


def contar_e_medir(marcadores_resultado, area_minima=20):
    """Conta as células e calcula a área (em pixels²) de cada uma,
    descartando regiões pequenas demais (provável ruído residual).

    A soma de pixels de cada rótulo é a versão discreta da integral
    dupla ∬ dA sobre a região da célula.
    """
    rotulos = np.unique(marcadores_resultado)
    # remove o fundo (1) e a linha de fronteira do watershed (-1)
    rotulos_celulas = [r for r in rotulos if r not in (1, -1)]

    resultados = []
    for rotulo in rotulos_celulas:
        area = int(np.sum(marcadores_resultado == rotulo))
        if area >= area_minima:
            resultados.append({"rotulo": int(rotulo), "area_px2": area})

    return resultados


def desenhar_resultado(img_original, marcadores_resultado, resultados):
    """Desenha as fronteiras detectadas (vermelho) e numera cada célula
    válida sobre a imagem original, para inspeção visual.
    """
    saida = img_original.copy()
    saida[marcadores_resultado == -1] = [0, 0, 255]  # fronteiras em vermelho

    for item in resultados:
        rotulo = item["rotulo"]
        mascara_celula = np.uint8(marcadores_resultado == rotulo) * 255
        M = cv2.moments(mascara_celula)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(
                saida, str(rotulo), (cx - 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
    return saida


def processar_imagem(caminho_imagem, fator_distancia=0.5, area_minima=20):
    """Executa o pipeline completo e retorna imagem final + estatísticas."""
    img_original, _, suave = carregar_e_preparar(caminho_imagem)
    binaria = binarizar(suave)
    limpa = limpar_ruido(binaria)
    marcadores, dist = gerar_marcadores(limpa, fator_distancia)
    marcadores_resultado = aplicar_watershed(img_original, marcadores)
    resultados = contar_e_medir(marcadores_resultado, area_minima)
    saida = desenhar_resultado(img_original, marcadores_resultado, resultados)

    areas = [r["area_px2"] for r in resultados]
    estatisticas = {
        "num_celulas": len(resultados),
        "area_media_px2": float(np.mean(areas)) if areas else 0.0,
        "area_desvio_px2": float(np.std(areas)) if areas else 0.0,
        "area_min_px2": float(np.min(areas)) if areas else 0.0,
        "area_max_px2": float(np.max(areas)) if areas else 0.0,
    }

    return saida, resultados, estatisticas, binaria, limpa, dist
