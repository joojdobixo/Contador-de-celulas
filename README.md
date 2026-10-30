# Rastreador de Contornos e Medição de Objetos

### Projeto de Extensão — Cálculo III (DIN00018) | UNIR/DACC 2026.2


---

## Tema

Aplicação de **Processamento de Imagens e Visão Computacional** que detecta o contorno de um objeto em uma imagem e calcula automaticamente sua **área**, **perímetro** e **centro de massa**, utilizando funções prontas da biblioteca OpenCV.

**Fundamentação matemática (Cálculo III):**

- Gradiente (∇f) → base da detecção de bordas (operador Sobel/Canny)
- Teorema de Green → base conceitual do cálculo de área de uma região delimitada por uma curva fechada (∮ (x dy − y dx)/2)
- Integral de linha (∮ ds) → base conceitual do cálculo de perímetro
- Integrais duplas (∬ x dA, ∬ y dA) → base conceitual do cálculo do centroide

> Mesmo usando as funções prontas da biblioteca, cada etapa do pipeline corresponde a um conceito específico de Cálculo III, que deve ser explicado no relatório e na apresentação.
> 

---

## Bibliotecas Utilizadas

| Biblioteca | Função no projeto |
| --- | --- |
| `opencv-python` (cv2) | Leitura de imagem, detecção de bordas, extração e medição de contornos |
| `numpy` | Manipulação de arrays e operações auxiliares |
| `matplotlib` | Visualização dos resultados (opcional, para o relatório) |

Instalação:

```bash
pip install opencv-python numpy matplotlib
```

---

## Passo a Passo (usando funções da biblioteca)

### Passo 1 — Carregar a imagem

```python
import cv2

img = cv2.imread("objeto.jpg")
img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

**Conceito relacionado:** conversão da imagem para função escalar f(x, y) de intensidade.

---

### Passo 2 — Suavizar a imagem (reduzir ruído)

```python
img_suave = cv2.GaussianBlur(img_cinza, (5, 5), 0)
```

**Conceito relacionado:** aproximação/suavização local, associada à ideia de série de Taylor para atenuar variações bruscas indesejadas (ruído).

---

### Passo 3 — Detectar bordas (gradiente)

```python
bordas = cv2.Canny(img_suave, 50, 150)
```

**Conceito relacionado:** o operador de Canny internamente calcula o **vetor gradiente** ∇f = (∂f/∂x, ∂f/∂y) e sua magnitude, identificando onde a intensidade muda bruscamente.

---

### Passo 4 — Extrair o contorno do objeto

```python
contornos, _ = cv2.findContours(
    bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
maior_contorno = max(contornos, key=cv2.contourArea)
```

**Conceito relacionado:** a curva fechada C resultante é o caminho de integração usado nas próximas etapas (Green e integral de linha).

---

### Passo 5 — Calcular área, perímetro e centroide

```python
area = cv2.contourArea(maior_contorno)
perimetro = cv2.arcLength(maior_contorno, closed=True)

momentos = cv2.moments(maior_contorno)
cx = int(momentos["m10"] / momentos["m00"])
cy = int(momentos["m01"] / momentos["m00"])
```

| Função da biblioteca | Conceito de Cálculo III correspondente |
| --- | --- |
| `cv2.contourArea()` | Teorema de Green: A = ∮ (x dy − y dx) / 2 |
| `cv2.arcLength()` | Integral de linha: ∮ ds |
| `cv2.moments()` | Integrais duplas de massa/centroide: x̄ = (1/A)∬x dA, ȳ = (1/A)∬y dA |

---

### Passo 6 — Desenhar o resultado na imagem

```python
img_resultado = img.copy()
cv2.drawContours(img_resultado, [maior_contorno], -1, (0, 255, 0), 2)
cv2.circle(img_resultado, (cx, cy), 5, (0, 0, 255), -1)

cv2.putText(img_resultado, f"Area: {area:.1f} px2", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(img_resultado, f"Perimetro: {perimetro:.1f} px", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
```

---

### Passo 7 — Exibir/salvar o resultado

```python
cv2.imshow("Resultado", img_resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("resultado.jpg", img_resultado)
```

---

## Resumo do Pipeline

```
Imagem original
   → escala de cinza
   → suavização (Gaussian Blur)
   → detecção de bordas (Canny → gradiente)
   → extração de contorno (findContours)
   → cálculo de área (contourArea → Teorema de Green)
   → cálculo de perímetro (arcLength → integral de linha)
   → cálculo de centroide (moments → integrais duplas)
   → visualização do resultado
```

---

## Objetivo da aplicação

O objetivo do projeto é criar um contador automático de células em imagens de microscopia, usando as bibliotecas OpenCV, NumPy e Matplotlib para identificar, separar e medir cada célula, aplicando conceitos de Cálculo III em uma ferramenta de código aberto para uso educacional.
