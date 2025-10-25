"""
TRADUCTOR.PY - Lógica de Traducción de Señas
MediaPipe + TensorFlow Lite para detectar y predecir señas
"""
import cv2
import numpy as np
import mediapipe as mp
import json
import os
import time

# Configurar TensorFlow (silenciar warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("❌ TensorFlow no disponible")
    TENSORFLOW_AVAILABLE = False

class SignLanguageTranslator:
    """Traductor de Lengua de Señas Colombiano"""
    
    def __init__(self):
        print("🔄 Inicializando traductor...")
        
        # Configuración mejorada
        self.MOVEMENT_THRESHOLD = 0.012  # Reducido para mejor sensibilidad
        self.STATIC_FRAMES_REQUIRED = 12  # Reducido para respuesta más rápida
        self.PREDICTION_COOLDOWN = 25
        self.DYNAMIC_MIN_FRAMES = 10
        self.MODEL_FRAMES = 40
        self.CONFIDENCE_STATIC = 0.65  # Umbral reducido
        self.CONFIDENCE_DYNAMIC = 0.55  # Umbral reducido
        
        # Estado
        self.sentence = []
        self.kp_sequence = []
        self.previous_kp = None
        self.static_counter = 0
        self.cooldown_counter = 0
        self.hands_present = False
        self.hands_were_present = False
        self.last_prediction_time = 0
        self.prediction_cooldown = 1.0  # 1 segundo entre predicciones
        
        # Modelos TFLite
        self.interpreter_static = None
        self.interpreter_dynamic = None
        self.input_details_static = None
        self.output_details_static = None
        self.input_details_dynamic = None
        self.output_details_dynamic = None
        
        # Estadísticas
        self.stats = {
            'static_predictions': 0,
            'dynamic_predictions': 0,
            'errors': 0
        }
        
        # Cargar todo
        self.load_models()
        self.load_words()
        self.init_mediapipe()
        
        print("✅ Traductor listo")
    
    def load_models(self):
        """Carga modelos TensorFlow Lite"""
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow no disponible - Modo simulación activado")
            return
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        models_loaded = 0
        
        # Modelo estático (letras) - TFLite
        static_path = os.path.join(base_path, "models", "static_letters_model.tflite")
        if os.path.exists(static_path):
            try:
                self.interpreter_static = tf.lite.Interpreter(model_path=static_path)
                self.interpreter_static.allocate_tensors()
                self.input_details_static = self.interpreter_static.get_input_details()
                self.output_details_static = self.interpreter_static.get_output_details()
                print("✅ Modelo letras TFLite cargado")
                models_loaded += 1
            except Exception as e:
                print(f"❌ Error cargando modelo estático: {e}")
        else:
            print(f"❌ Modelo estático no encontrado: {static_path}")
        
        # Modelo dinámico (palabras) - TFLite
        dynamic_path = os.path.join(base_path, "models", "actions_40.tflite")
        if os.path.exists(dynamic_path):
            try:
                print(f"📂 Cargando modelo dinámico desde: {dynamic_path}")
                self.interpreter_dynamic = tf.lite.Interpreter(model_path=dynamic_path)
                self.interpreter_dynamic.allocate_tensors()
                self.input_details_dynamic = self.interpreter_dynamic.get_input_details()
                self.output_details_dynamic = self.interpreter_dynamic.get_output_details()
                
                # Verificar que se cargaron correctamente
                if self.input_details_dynamic and self.output_details_dynamic:
                    print(f"✅ Modelo palabras TFLite cargado")
                    print(f"   Input shape: {self.input_details_dynamic[0]['shape']}")
                    print(f"   Output shape: {self.output_details_dynamic[0]['shape']}")
                    models_loaded += 1
                else:
                    print("⚠️ Modelo cargado pero sin detalles de input/output")
                    self.interpreter_dynamic = None
            except RuntimeError as e:
                if "Select TensorFlow op" in str(e):
                    print("⚠️ Modelo dinámico requiere SELECT_TF_OPS (funcionará en Android)")
                    print("   En PC, solo funcionarán las letras estáticas")
                else:
                    print(f"❌ Error cargando modelo dinámico: {e}")
                self.interpreter_dynamic = None
                self.input_details_dynamic = None
                self.output_details_dynamic = None
            except Exception as e:
                print(f"❌ Error cargando modelo dinámico: {e}")
                import traceback
                traceback.print_exc()
                self.interpreter_dynamic = None
                self.input_details_dynamic = None
                self.output_details_dynamic = None
        else:
            print(f"❌ Modelo dinámico no encontrado: {dynamic_path}")
        
        if models_loaded == 0:
            print("⚠️ No se cargaron modelos - Verifica la carpeta 'models'")
    
    def load_words(self):
        """Carga letras y palabras"""
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            
            # Cargar words.json
            words_json_path = os.path.join(base_path, "words.json")
            if os.path.exists(words_json_path):
                with open(words_json_path, "r", encoding="utf-8") as f:
                    words_data = json.load(f)
                
                self.word_ids = words_data.get("word_ids", [])
                self.word_static = words_data.get("word_static", [])
                print(f"✅ {len(self.word_static)} letras, {len(self.word_ids)} palabras cargadas")
            else:
                print("❌ words.json no encontrado, usando valores por defecto")
                self.word_static = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", 
                                  "K", "L", "M", "N", "Ñ", "O", "P", "Q", "R", "S", 
                                  "T", "U", "V", "W", "X", "Y", "Z"]
                self.word_ids = ["hola", "gracias", "por_favor", "adios", "si", "no"]
            
            # Cargar diccionario de palabras
            dict_path = os.path.join(base_path, "words_dict.json")
            if os.path.exists(dict_path):
                with open(dict_path, "r", encoding="utf-8") as f:
                    self.words_text = json.load(f)
            else:
                print("❌ words_dict.json no encontrado, usando diccionario básico")
                self.words_text = {
                    "hola": "HOLA", 
                    "gracias": "GRACIAS",
                    "por_favor": "POR FAVOR", 
                    "adios": "ADIÓS",
                    "si": "SÍ", 
                    "no": "NO",
                    "como_estas": "¿CÓMO ESTÁS?",
                    "bien": "BIEN",
                    "mal": "MAL",
                    "nombre": "NOMBRE"
                }
            
        except Exception as e:
            print(f"❌ Error cargando vocabulario: {e}")
            # Valores de respaldo
            self.word_static = ["A", "B", "C"]
            self.word_ids = ["hola", "gracias"]
            self.words_text = {"hola": "HOLA", "gracias": "GRACIAS"}
    
    def init_mediapipe(self):
        """Inicia MediaPipe"""
        try:
            mp_holistic = mp.solutions.holistic
            self.holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.6,  # Reducido para mejor detección
                min_tracking_confidence=0.5
            )
            print("✅ MediaPipe inicializado correctamente")
        except Exception as e:
            print(f"❌ Error inicializando MediaPipe: {e}")
            self.holistic = None
    
    def extract_keypoints(self, results):
        """Extrae 258 características (Pose + Manos)"""
        try:
            # Pose: 33 × 4 = 132
            pose = np.array([[r.x, r.y, r.z, r.visibility] 
                            for r in results.pose_landmarks.landmark]).flatten() \
                   if results.pose_landmarks else np.zeros(132)
            
            # Mano izquierda: 21 × 3 = 63
            lh = np.array([[r.x, r.y, r.z] 
                          for r in results.left_hand_landmarks.landmark]).flatten() \
                 if results.left_hand_landmarks else np.zeros(63)
            
            # Mano derecha: 21 × 3 = 63
            rh = np.array([[r.x, r.y, r.z] 
                          for r in results.right_hand_landmarks.landmark]).flatten() \
                 if results.right_hand_landmarks else np.zeros(63)
            
            return np.concatenate([pose, lh, rh])
        except Exception as e:
            print(f"❌ Error extrayendo keypoints: {e}")
            return np.zeros(258)  # Array vacío si hay error
    
    def extract_hand_keypoints_static(self, kp_frame):
        """Extrae y normaliza una sola mano para letras"""
        try:
            mano_izq = kp_frame[132:195]  # 63 puntos (21 landmarks × 3 coordenadas)
            mano_der = kp_frame[195:258]  # 63 puntos
            
            # Preferir mano derecha, si no hay, usar izquierda (espejada)
            if np.sum(np.abs(mano_der)) > 0.1:  # Umbral de detección
                return mano_der.astype("float32")
            elif np.sum(np.abs(mano_izq)) > 0.1:
                # Espejar mano izquierda para consistencia
                mano = mano_izq.reshape((21, 3))
                mano[:, 0] = 1 - mano[:, 0]  # Invertir coordenada X
                return mano.flatten().astype("float32")
            else:
                return np.zeros(63, dtype="float32")
        except Exception as e:
            print(f"❌ Error extrayendo mano estática: {e}")
            return np.zeros(63, dtype="float32")
    
    def calculate_movement(self, kp_current, kp_previous):
        """Calcula movimiento entre frames"""
        if kp_previous is None:
            return 1.0  # Máximo movimiento en el primer frame
        
        try:
            # Calcular diferencia solo en puntos significativos
            if len(kp_current) == len(kp_previous):
                # Usar solo puntos de manos (excluir pose para mejor sensibilidad)
                hands_current = kp_current[132:]  # Solo manos
                hands_previous = kp_previous[132:]  # Solo manos
                
                non_zero_mask = (np.abs(hands_previous) > 0.01) | (np.abs(hands_current) > 0.01)
                if np.any(non_zero_mask):
                    movement = np.mean(np.abs(hands_current[non_zero_mask] - hands_previous[non_zero_mask]))
                    return movement
                else:
                    return 0.0
            return 1.0
        except Exception as e:
            print(f"❌ Error calculando movimiento: {e}")
            return 1.0
    
    def normalize_keypoints(self, keypoints, target_length=40):
        """Normaliza secuencia a longitud objetivo"""
        if len(keypoints) == 0:
            return np.zeros((target_length, 258))
        
        current_length = len(keypoints)
        
        if current_length == target_length:
            return np.array(keypoints)
        elif current_length < target_length:
            # Repetir el último frame para alcanzar la longitud
            repeated = [keypoints[-1]] * (target_length - current_length)
            return np.array(keypoints + repeated)
        else:
            # Submuestreo uniforme
            indices = np.linspace(0, current_length - 1, target_length).astype(int)
            return np.array([keypoints[i] for i in indices])
    
    def process_frame(self, frame):
        """
        Procesa un frame de la cámara
        Retorna: dict con status, prediction, type
        """
        try:
            current_time = time.time()
            
            # Verificar cooldown
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
            
            # Verificar MediaPipe
            if self.holistic is None:
                return {"status": "error", "message": "MediaPipe no inicializado"}
            
            # Procesar con MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            
            # Detectar presencia de manos
            hands_detected = (results.left_hand_landmarks is not None or 
                            results.right_hand_landmarks is not None)
            
            # === SIN MANOS VISIBLES ===
            if not hands_detected:
                self.static_counter = 0
                
                # Si acababan de haber manos, procesar secuencia dinámica
                if (self.hands_were_present and 
                    len(self.kp_sequence) >= self.DYNAMIC_MIN_FRAMES and 
                    self.cooldown_counter == 0 and
                    current_time - self.last_prediction_time > self.prediction_cooldown):
                    
                    prediction = self.predict_dynamic()
                    
                    if prediction:
                        self.sentence.append(prediction)
                        self.cooldown_counter = self.PREDICTION_COOLDOWN
                        self.last_prediction_time = current_time
                        self.stats['dynamic_predictions'] += 1
                        
                        # Limpiar para siguiente detección
                        self.kp_sequence = []
                        self.previous_kp = None
                        self.hands_were_present = False
                        
                        return {
                            "prediction": prediction, 
                            "type": "dynamic", 
                            "status": "success",
                            "confidence": "high"
                        }
                
                self.hands_were_present = False
                self.kp_sequence = []
                self.previous_kp = None
                return {"status": "waiting", "message": "Muestra las manos en la cámara"}
            
            # === CON MANOS VISIBLES ===
            self.hands_were_present = True
            
            # Extraer keypoints
            kp_frame = self.extract_keypoints(results)
            
            # Calcular movimiento
            movement = self.calculate_movement(kp_frame, self.previous_kp)
            self.previous_kp = kp_frame.copy()
            
            # Actualizar contadores de estabilidad
            if movement < self.MOVEMENT_THRESHOLD:
                self.static_counter += 1
            else:
                self.static_counter = max(0, self.static_counter - 2)  # Penalizar movimiento
            
            # Acumular para detección dinámica
            self.kp_sequence.append(kp_frame)
            if len(self.kp_sequence) > 60:  # Limitar longitud
                self.kp_sequence = self.kp_sequence[-60:]
            
            # PREDICCIÓN ESTÁTICA (Letras)
            if (self.static_counter >= self.STATIC_FRAMES_REQUIRED and 
                self.cooldown_counter == 0 and
                current_time - self.last_prediction_time > self.prediction_cooldown):
                
                prediction = self.predict_static(kp_frame)
                
                if prediction:
                    self.sentence.append(prediction)
                    self.cooldown_counter = self.PREDICTION_COOLDOWN
                    self.last_prediction_time = current_time
                    self.static_counter = 0  # Resetear contador
                    self.stats['static_predictions'] += 1
                    
                    return {
                        "prediction": prediction, 
                        "type": "static", 
                        "status": "success",
                        "confidence": "high"
                    }
            
            # Proporcionar feedback visual
            if self.static_counter > 0:
                progress = min(100, (self.static_counter / self.STATIC_FRAMES_REQUIRED) * 100)
                return {
                    "status": "accumulating",
                    "counter": f"{self.static_counter}/{self.STATIC_FRAMES_REQUIRED}",
                    "movement": f"{movement:.4f}",
                    "progress": progress,
                    "message": "Mantén la mano quieta para letras"
                }
            else:
                return {
                    "status": "detecting",
                    "movement": f"{movement:.4f}",
                    "message": "Mueve las manos para palabras",
                    "sequence_length": len(self.kp_sequence)
                }
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"❌ Error en process_frame: {e}")
            return {"status": "error", "message": f"Error: {str(e)}"}
    
    def predict_static(self, kp_frame):
        """Predice letra estática usando TFLite"""
        try:
            if self.interpreter_static is None:
                print("⚠️ Modelo estático no disponible")
                return None
            
            # Extraer y preparar datos de mano
            mano_data = self.extract_hand_keypoints_static(kp_frame)
            
            # Verificar que hay datos válidos
            if np.sum(np.abs(mano_data)) < 0.05:
                return None
            
            # Normalizar
            max_val = np.max(np.abs(mano_data))
            if max_val > 0:
                normalized_data = mano_data / max_val
            else:
                normalized_data = mano_data
            
            # Preparar input para TFLite
            input_data = np.expand_dims(normalized_data, axis=0).astype(np.float32)
            
            # Predicción con TFLite
            self.interpreter_static.set_tensor(self.input_details_static[0]['index'], input_data)
            self.interpreter_static.invoke()
            predictions = self.interpreter_static.get_tensor(self.output_details_static[0]['index'])[0]
            
            pred_idx = np.argmax(predictions)
            confidence = float(predictions[pred_idx])
            
            # Verificar umbral de confianza
            if confidence >= self.CONFIDENCE_STATIC and pred_idx < len(self.word_static):
                letter = self.word_static[pred_idx]
                print(f"🔤 Letra detectada: {letter} ({confidence*100:.1f}%)")
                return letter
            else:
                if pred_idx < len(self.word_static):
                    print(f"⚠️ Letra baja confianza: {self.word_static[pred_idx]} ({confidence*100:.1f}%)")
                return None
            
        except Exception as e:
            print(f"❌ Error en predict_static: {e}")
            return None
    
    def predict_dynamic(self):
        """Predice palabra dinámica usando TFLite"""
        try:
            # Verificar que el modelo y sus detalles existan
            if (self.interpreter_dynamic is None or 
                self.input_details_dynamic is None or
                self.output_details_dynamic is None or
                len(self.kp_sequence) < self.DYNAMIC_MIN_FRAMES):
                return None
            
            # Normalizar secuencia
            kp_normalized = self.normalize_keypoints(self.kp_sequence, self.MODEL_FRAMES)
            
            # Preparar input para TFLite
            input_data = np.expand_dims(kp_normalized, axis=0).astype(np.float32)
            
            # Predicción con TFLite
            self.interpreter_dynamic.set_tensor(self.input_details_dynamic[0]['index'], input_data)
            self.interpreter_dynamic.invoke()
            predictions = self.interpreter_dynamic.get_tensor(self.output_details_dynamic[0]['index'])[0]
            
            pred_idx = np.argmax(predictions)
            confidence = float(predictions[pred_idx])
            
            # Obtener palabra
            if pred_idx < len(self.word_ids):
                word_id = self.word_ids[pred_idx].split('-')[0]
                word_name = self.words_text.get(word_id, word_id.upper())
            else:
                word_name = f"Palabra_{pred_idx}"
            
            # Verificar confianza
            if confidence >= self.CONFIDENCE_DYNAMIC:
                print(f"👋 Palabra detectada: {word_name} ({confidence*100:.1f}%)")
                return word_name
            else:
                print(f"⚠️ Palabra baja confianza: {word_name} ({confidence*100:.1f}%)")
                return None
            
        except Exception as e:
            print(f"❌ Error en predict_dynamic: {e}")
            return None
    
    def reset(self):
        """Reinicia el estado del traductor"""
        self.sentence = []
        self.kp_sequence = []
        self.previous_kp = None
        self.static_counter = 0
        self.cooldown_counter = 0
        self.hands_were_present = False
        self.last_prediction_time = 0
        print("🔄 Traductor reiniciado")
    
    def get_stats(self):
        """Retorna estadísticas de uso"""
        return self.stats
    
    def get_sentence(self):
        """Retorna la oración actual"""
        return ' '.join(self.sentence)