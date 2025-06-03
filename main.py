import os
import cv2
from ultralytics import YOLO
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt  # Solo si necesitas visualizar mapas de profundidad por separado
import torch  # Para MiDaS/DPT

print("Paso 0: Librerías importadas correctamente.")

# ==============================================================================
# PASO 1: CONFIGURACIÓN DE RUTAS Y PARÁMETROS GLOBALES
# ==============================================================================
print("\nPaso 1: Configurando rutas y parámetros...")

MODEL_PATH_VEHICLES = 'F:/Documents/PycharmProjects/MiDaSDetector/best.pt'  # TU MODELO DE VEHÍCULOS
VIDEO_INPUT_PATH = 'F:\Documents\PycharmProjects\MiDaSDetector\GH012372_no_audio.mp4'  # VIDEO SIN AUDIO O RE-CODIFICADO

OUTPUT_DIR_ADVANCED = "./runs_local/advanced_video_processing"
os.makedirs(OUTPUT_DIR_ADVANCED, exist_ok=True)
VIDEO_BASENAME_ADV = os.path.splitext(os.path.basename(VIDEO_INPUT_PATH))[0]
VIDEO_OUTPUT_PATH_ADVANCED = os.path.join(OUTPUT_DIR_ADVANCED, f"{VIDEO_BASENAME_ADV}_annotated_depth_distance.mp4")

print(f"Modelo de vehículos: {os.path.abspath(MODEL_PATH_VEHICLES)}")
print(f"Video de entrada: {os.path.abspath(VIDEO_INPUT_PATH)}")
print(f"Video de salida se guardará en: {os.path.abspath(VIDEO_OUTPUT_PATH_ADVANCED)}")

# Parámetros de inferencia YOLO
CONFIDENCE_THRESHOLD = 0.35

# Parámetros para estimación de distancia (fallback y MiDaS)
FOCAL_LENGTH_PX = 700  # <<<< AJUSTA ESTO - CALIBRACIÓN DE CÁMARA!
REAL_OBJECT_SIZES_M = {
    'car': 1.8, 'threewheel': 1.2, 'bus': 2.5, 'truck': 2.6, 'motorbike': 0.8, 'van': 2.0,
    'person': 0.5, 'bicycle': 0.4, 'dog': 0.3
}  # <<<< AJUSTA ESTOS VALORES!

COCO_CLASSES_TO_SEEK = ['person', 'bicycle', 'dog']

# ==============================================================================
# PASO 2: CARGAR TODOS LOS MODELOS
# ==============================================================================
print("\nPaso 2: Cargando modelos...")

# --- Modelo de Vehículos (Tu fine-tuned) ---
if not os.path.exists(MODEL_PATH_VEHICLES):
    raise FileNotFoundError(f"Modelo de vehículos no encontrado en: {MODEL_PATH_VEHICLES}")
model_vehicles = YOLO(MODEL_PATH_VEHICLES)
print(f"Modelo de vehículos '{MODEL_PATH_VEHICLES}' cargado. Clases: {model_vehicles.names}")

# --- Modelo COCO (YOLOv8n pre-entrenado) ---
model_coco = YOLO('yolov8n.pt')
print(f"Modelo COCO 'yolov8n.pt' cargado.")
coco_target_ids = []
if isinstance(model_coco.names, dict):
    for name_to_seek in COCO_CLASSES_TO_SEEK:
        found_id = None
        for class_id_coco, class_name_coco in model_coco.names.items():
            if class_name_coco == name_to_seek: found_id = class_id_coco; break
        if found_id is not None:
            coco_target_ids.append(found_id)
        else:
            print(f"ADVERTENCIA: Clase COCO '{name_to_seek}' no encontrada.")
else:  # Fallback (raro)
    for name_to_seek in COCO_CLASSES_TO_SEEK:
        try:
            coco_target_ids.append(model_coco.names.index(name_to_seek))
        except ValueError:
            print(f"ADVERTENCIA: Clase COCO '{name_to_seek}' no encontrada (names como lista).")
print(f"IDs de clases COCO a detectar: {coco_target_ids}")

# --- Modelo de Estimación de Profundidad (MiDaS) ---
print("\nCargando modelo de estimación de profundidad (MiDaS)...")
depth_model_loaded_successfully = False
midas_model = None
midas_transform = None
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

try:
    model_type_depth = "MiDaS_small"  # Opciones: "MiDaS", "dpt_beit_large_512", etc.
    midas_model = torch.hub.load("intel-isl/MiDaS", model_type_depth,
                                 trust_repo=True)  # trust_repo=True puede ser necesario

    if "dpt" in model_type_depth.lower() or "beit" in model_type_depth.lower():
        midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).dpt_transform
    else:
        midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).midas_transform

    midas_model.to(device)
    midas_model.eval()
    print(f"Modelo de profundidad '{model_type_depth}' cargado y en dispositivo '{device}'.")
    depth_model_loaded_successfully = True
except Exception as e_load_depth:
    print(f"Error al cargar el modelo de profundidad MiDaS: {e_load_depth}")
    print("El procesamiento continuará sin estimación de profundidad avanzada (usará fallback).")

# ==============================================================================
# PASO 3: INICIAR PROCESAMIENTO DE VIDEO MANUAL
# ==============================================================================
print("\nPaso 3: Iniciando procesamiento manual del video con estimación de profundidad...")

if not os.path.exists(VIDEO_INPUT_PATH):
    raise FileNotFoundError(f"Video de entrada no encontrado en: {VIDEO_INPUT_PATH}")

cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
if not cap.isOpened():
    raise IOError(f"No se pudo abrir el video de entrada: {VIDEO_INPUT_PATH}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)
total_frames_input = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_output = fps_input if fps_input > 0 else 30.0
print(f"Video de entrada: {frame_width}x{frame_height} @ {fps_input:.2f} FPS, Total Frames: {total_frames_input}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(VIDEO_OUTPUT_PATH_ADVANCED, fourcc, fps_output, (frame_width, frame_height))
if not out_video.isOpened():
    cap.release()
    raise IOError(f"No se pudo abrir el VideoWriter para: {VIDEO_OUTPUT_PATH_ADVANCED}")

print(f"Procesando video y guardando en: {VIDEO_OUTPUT_PATH_ADVANCED}")
frames_processed_count = 0
frames_written_count = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Fin del video o error de lectura en el frame {frames_processed_count + 1}.")
            break

        frames_processed_count += 1
        annotated_frame = frame.copy()
        all_detections_current_frame = []

        # --- 1. Inferencia con modelo de vehículos ---
        results_v = model_vehicles.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        if results_v:
            for box_v_data in results_v[0].boxes:
                all_detections_current_frame.append({
                    "xywh": box_v_data.xywh.cpu().numpy()[0], "cls_id": int(box_v_data.cls.cpu().item()),
                    "conf": float(box_v_data.conf.cpu().item()), "model_names_map": model_vehicles.names,
                    "is_vehicle": True
                })

        # --- 2. Inferencia con modelo COCO ---
        if coco_target_ids:
            results_c = model_coco.predict(source=frame, conf=CONFIDENCE_THRESHOLD, classes=coco_target_ids,
                                           verbose=False)
            if results_c:
                for box_c_data in results_c[0].boxes:
                    all_detections_current_frame.append({
                        "xywh": box_c_data.xywh.cpu().numpy()[0], "cls_id": int(box_c_data.cls.cpu().item()),
                        "conf": float(box_c_data.conf.cpu().item()), "model_names_map": model_coco.names,
                        "is_vehicle": False
                    })

        # --- 3. Estimación del Mapa de Profundidad con MiDaS ---
        depth_map_values = None
        if depth_model_loaded_successfully and midas_model and midas_transform:
            try:
                img_rgb_for_depth = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_batch_depth = midas_transform(img_rgb_for_depth).to(device)
                with torch.no_grad():
                    prediction_depth = midas_model(input_batch_depth)
                    prediction_depth = torch.nn.functional.interpolate(
                        prediction_depth.unsqueeze(1), size=img_rgb_for_depth.shape[:2],
                        mode="bicubic", align_corners=False,
                    ).squeeze()
                depth_map_values = prediction_depth.cpu().numpy()
            except Exception as e_depth_frame:
                # print(f"Advertencia: Error procesando profundidad para frame {frames_processed_count}: {e_depth_frame}")
                depth_map_values = None

        # --- 4. Dibujar todas las detecciones y estimar distancia ---
        for det in all_detections_current_frame:
            x_c, y_c, w, h = det["xywh"];
            cls_id = det["cls_id"];
            conf_val = det["conf"]
            current_model_names = det["model_names_map"];
            label_name = current_model_names[cls_id]
            x1, y1, x2, y2 = int(x_c - w / 2), int(y_c - h / 2), int(x_c + w / 2), int(y_c + h / 2)
            box_clr = (0, 255, 0) if det["is_vehicle"] else (255, 165, 0)  # Verde para vehículos, Naranja para COCO
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_clr, 2)

            dist_m_final = -1
            dist_source = "N/A"  # Para saber de dónde vino la estimación

            if depth_map_values is not None:
                roi_y1, roi_y2 = np.clip(y1, 0, depth_map_values.shape[0] - 1), np.clip(y2, 0,
                                                                                        depth_map_values.shape[0] - 1)
                roi_x1, roi_x2 = np.clip(x1, 0, depth_map_values.shape[1] - 1), np.clip(x2, 0,
                                                                                        depth_map_values.shape[1] - 1)
                if roi_y1 < roi_y2 and roi_x1 < roi_x2:
                    roi_depth_patch = depth_map_values[roi_y1:roi_y2, roi_x1:roi_x2]
                    if roi_depth_patch.size > 0:
                        median_depth_val_map = np.median(roi_depth_patch)
                        # ---- ¡¡¡PLACEHOLDER MUY IMPORTANTE PARA ESCALA DE MiDaS!!! ----
                        # Necesitas reemplazar esto con una calibración/escalado adecuado.
                        # Esto es solo un ejemplo y NO dará metros precisos sin más.
                        # Si MiDaS da disparidad (valores altos = más cerca):
                        if median_depth_val_map > 1e-4:  # Evitar división por cero/infinito
                            # El factor '30.0' es COMPLETAMENTE ARBITRARIO aquí.
                            # Podría ser 1.0, 100.0, o necesitar una función más compleja.
                            dist_m_final = 30.0 / median_depth_val_map
                            dist_m_final = max(0.1, min(dist_m_final, 150.0))  # Limitar a un rango plausible
                            dist_source = "MiDaS"
                        # -------------------------------------------------------------

            if dist_m_final <= 0 and label_name in REAL_OBJECT_SIZES_M and REAL_OBJECT_SIZES_M[label_name] > 0:
                w_real_m = REAL_OBJECT_SIZES_M[label_name]
                if w > 0: dist_m_final = (w_real_m * FOCAL_LENGTH_PX) / w
                dist_source = "Size"

            txt_lbl = f"{label_name} {conf_val:.2f}"
            if dist_m_final > 0: txt_lbl += f" {dist_m_final:.1f}m ({dist_source[0]})"  # Añadir (M) o (S)

            (txt_w, txt_h), _ = cv2.getTextSize(txt_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)  # Fuente más pequeña
            cv2.rectangle(annotated_frame, (x1, y1 - txt_h - 3), (x1 + txt_w, y1 - 1), box_clr, -1)
            cv2.putText(annotated_frame, txt_lbl, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                        cv2.LINE_AA)

        out_video.write(annotated_frame)
        frames_written_count += 1

        if frames_processed_count % (int(fps_output) * 10) == 0:  # Progreso cada ~10 segundos
            print(f"  Procesado y escrito frame {frames_processed_count}/{total_frames_input}...")

except KeyboardInterrupt:  # Permitir detener con Ctrl+C
    print("\nProcesamiento interrumpido por el usuario.")
except Exception as e_proc_manual_adv:
    print(f"Error durante el procesamiento manual avanzado del video: {e_proc_manual_adv}")
    import traceback

    traceback.print_exc()
finally:
    print(f"\nCerrando archivos de video...")
    if cap.isOpened(): cap.release()
    if out_video.isOpened(): out_video.release()
    cv2.destroyAllWindows()

print(f"\nProcesamiento manual avanzado de video completado.")
print(f"Total frames leídos: {frames_processed_count}")
print(f"Total frames escritos: {frames_written_count}")

if os.path.exists(VIDEO_OUTPUT_PATH_ADVANCED) and frames_written_count > 0:
    print(f"\n✅ Video procesado guardado en: {os.path.abspath(VIDEO_OUTPUT_PATH_ADVANCED)}")
    try:  # Verificar duración
        cap_out_chk_adv = cv2.VideoCapture(VIDEO_OUTPUT_PATH_ADVANCED)
        if cap_out_chk_adv.isOpened():
            fps_out_adv = cap_out_chk_adv.get(cv2.CAP_PROP_FPS)
            fc_out_adv = int(cap_out_chk_adv.get(cv2.CAP_PROP_FRAME_COUNT))
            dur_s_out_adv = fc_out_adv / fps_out_adv if fps_out_adv > 0 else 0
            print(
                f"   Verificación salida: Frames: {fc_out_adv}, Duración: ~{dur_s_out_adv:.2f}s @ {fps_out_adv:.2f} FPS")
            cap_out_chk_adv.release()
    except Exception as e_verif_adv:
        print(f"Error verificando video de salida: {e_verif_adv}")
else:
    print("\n⚠️ No se guardó el video de salida o no se escribieron frames.")

print("\n--- Proceso de Inferencia de Video Avanzado Local Finalizado ---")
