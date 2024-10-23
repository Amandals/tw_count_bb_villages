import gradio as gr
import cv2
import numpy as np
import requests

def process_image(image_url):
    response = requests.get(image_url)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lower_barb = np.array([100, 100, 100])  # Ajuste conforme necessário
    upper_barb = np.array([200, 200, 200])  # Ajuste conforme necessário

    mask = cv2.inRange(img, lower_barb, upper_barb)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    numero_areas = len(contours)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    return img, numero_areas

def interface(image_url):
    processed_image, count = process_image(image_url)
    
    # Converter a imagem processada para formato que Gradio aceita
    _, buffer = cv2.imencode('.jpg', processed_image)
    return (buffer.tobytes(), f"Número de áreas 'bárbaras' encontradas: {count}")

# Criar a interface Gradio
iface = gr.Interface(
    fn=interface,
    inputs="text",
    outputs=["image", "text"],
    title="Contar Áreas 'Bárbaras' na Imagem",
    description="Insira a URL de uma imagem e clique em 'Submit' para calcular."
)

# Executar a aplicação
iface.launch()
