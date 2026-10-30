"""
Demo do Contador de Células via Watershed.

Se você já tem uma imagem real de microscopia, rode:
    python demo.py caminho/para/imagem.jpg

Se não tiver nenhuma imagem ainda, rode sem argumento:
    python demo.py
Isso gera uma imagem sintética com "células" (círculos) se sobrepondo,
para validar que o pipeline funciona antes de usar dados reais.
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from celulas_watershed import processar_imagem


def gerar_imagem_sintetica(caminho_saida="celulas_sinteticas.png",
                             largura=500, altura=500, num_circulos=18, seed=42):
    """Cria uma imagem com círculos brancos (simulando células) sobre fundo
    escuro, alguns propositalmente sobrepostos, para testar a separação
    do Watershed antes de partir para imagens reais de microscopia.
    """
    # Fundo claro e "células" mais escuras — mesmo padrão de contraste
    # de uma lâmina de microscopia com coloração (núcleos escuros sobre
    # fundo claro), que é o cenário assumido pelo threshold do pipeline.
    rng = np.random.default_rng(seed)
    img = np.zeros((altura, largura, 3), dtype=np.uint8)
    img[:] = (220, 220, 220)  # fundo claro

    for _ in range(num_circulos):
        raio = int(rng.integers(15, 35))
        cx = int(rng.integers(raio, largura - raio))
        cy = int(rng.integers(raio, altura - raio))
        cor = (70, 70, 70)
        cv2.circle(img, (cx, cy), raio, cor, -1)

    # um pouco de ruído para simular imperfeição de microscópio real
    ruido = rng.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

    cv2.imwrite(caminho_saida, img)
    return caminho_saida


def main():
    if len(sys.argv) > 1:
        caminho_imagem = sys.argv[1]
        print(f"Usando imagem fornecida: {caminho_imagem}")
    else:
        print("Nenhuma imagem fornecida — gerando imagem sintética de teste...")
        caminho_imagem = gerar_imagem_sintetica()
        print(f"Imagem sintética salva em: {caminho_imagem}")

    saida, resultados, estatisticas, binaria, limpa, dist = processar_imagem(
        caminho_imagem, fator_distancia=0.5, area_minima=20
    )

    print("\n=== RESULTADO ===")
    print(f"Número de células detectadas: {estatisticas['num_celulas']}")
    print(f"Área média: {estatisticas['area_media_px2']:.1f} px²")
    print(f"Desvio padrão da área: {estatisticas['area_desvio_px2']:.1f} px²")
    print(f"Área mínima: {estatisticas['area_min_px2']:.1f} px²")
    print(f"Área máxima: {estatisticas['area_max_px2']:.1f} px²")

    cv2.imwrite("resultado_final.png", saida)
    print("\nImagem com o resultado salva em: resultado_final.png")

    # painel com todas as etapas do pipeline, para o artigo/apresentação
    fig, eixos = plt.subplots(2, 3, figsize=(15, 9))

    img_original = cv2.imread(caminho_imagem)
    eixos[0, 0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    eixos[0, 0].set_title("1. Imagem original")

    eixos[0, 1].imshow(binaria, cmap="gray")
    eixos[0, 1].set_title("2. Binarização (Otsu)")

    eixos[0, 2].imshow(limpa, cmap="gray")
    eixos[0, 2].set_title("3. Limpeza morfológica")

    eixos[1, 0].imshow(dist, cmap="jet")
    eixos[1, 0].set_title("4. Distance Transform\n(campo escalar de distância)")

    eixos[1, 1].imshow(cv2.cvtColor(saida, cv2.COLOR_BGR2RGB))
    eixos[1, 1].set_title(f"5. Watershed final\n({estatisticas['num_celulas']} células)")

    areas = [r["area_px2"] for r in resultados]
    eixos[1, 2].hist(areas, bins=10, color="steelblue", edgecolor="black")
    eixos[1, 2].set_title("6. Histograma de áreas")
    eixos[1, 2].set_xlabel("Área (px²)")
    eixos[1, 2].set_ylabel("Quantidade de células")

    for linha in eixos:
        for eixo in linha:
            if eixo != eixos[1, 2]:
                eixo.axis("off")

    plt.tight_layout()
    plt.savefig("pipeline_completo.png", dpi=150)
    print("Painel com todas as etapas salvo em: pipeline_completo.png")
    plt.show()


if __name__ == "__main__":
    main()
