"""
PANTALLA_SENAS.PY - Señas a Texto
Aplica la nueva paleta de colores
"""
from kivy.uix.screenmanager import Screen, SlideTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.uix.progressbar import ProgressBar
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
import cv2
import pyttsx3
import time

# Importar el traductor
from traductor import SignLanguageTranslator

class SenaScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Configurar fondo blanco
        Window.clearcolor = (1, 1, 1, 1)
        
        # Inicializar componentes
        self.translator = SignLanguageTranslator()
        self.camera = None
        self.is_recording = False
        self.event = None
        self.update_interval = 1.0 / 15.0
        self.is_speaking = False
        
        # Configurar TTS
        self.tts_engine = self.init_tts()
        
        # Crear interfaz
        self.build_ui()
        
        print("✅ Pantalla de señas inicializada")
    
    def init_tts(self):
        """Inicializa text-to-speech"""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.setProperty('volume', 0.8)
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'spanish' in voice.name.lower() or 'español' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            return engine
        except Exception as e:
            print(f"⚠️ Text-to-speech no disponible: {e}")
            return None
    
    def build_ui(self):
        """Construye la interfaz de usuario con nueva paleta"""
        layout = BoxLayout(orientation='vertical', spacing=10, padding=20)
        
        # === HEADER ===
        header = BoxLayout(size_hint=(1, 0.08), spacing=10)
        btn_back = Button(
            text='←', 
            size_hint=(0.15, 1), 
            background_color=(0.12, 0.23, 0.54, 1),  # #1E3A8A
            background_normal='',
            font_size='20sp',
            color=(1, 1, 1, 1)
        )
        btn_back.bind(on_press=self.volver)
        header.add_widget(btn_back)
        
        header_label = Label(
            text='Señas a Texto', 
            font_size='24sp', 
            bold=True, 
            size_hint=(0.85, 1),
            color=(0.12, 0.23, 0.54, 1)  # #1E3A8A
        )
        header.add_widget(header_label)
        layout.add_widget(header)
        
        # === CÁMARA ===
        cam_container = BoxLayout(orientation='vertical', size_hint=(1, 0.45), spacing=5)
        cam_container.add_widget(Label(
            text='Vista de la cámara',
            size_hint=(1, 0.1),
            font_size='14sp',
            color=(0.4, 0.4, 0.4, 1)
        ))
        
        self.img_widget = KivyImage(
            size_hint=(1, 0.9),
            allow_stretch=True,
            keep_ratio=True
        )
        cam_container.add_widget(self.img_widget)
        layout.add_widget(cam_container)
        
        # === BARRA DE PROGRESO ===
        progress_container = BoxLayout(orientation='vertical', size_hint=(1, 0.05), spacing=2)
        progress_container.add_widget(Label(
            text='Progreso de detección',
            size_hint=(1, 0.4),
            font_size='12sp',
            color=(0.6, 0.6, 0.6, 1)
        ))
        
        self.progress_bar = ProgressBar(
            size_hint=(1, 0.6),
            max=100,
            value=0
        )
        progress_container.add_widget(self.progress_bar)
        layout.add_widget(progress_container)
        
        # === ESTADO DE DETECCIÓN ===
        self.prediction_label = Label(
            text='Presiona INICIAR para comenzar',
            size_hint=(1, 0.08),
            font_size='16sp',
            color=(0.12, 0.23, 0.54, 1),  # #1E3A8A
            bold=True
        )
        layout.add_widget(self.prediction_label)
        
        # === INFORMACIÓN ADICIONAL ===
        self.info_label = Label(
            text='',
            size_hint=(1, 0.04),
            font_size='12sp',
            color=(0.6, 0.6, 0.6, 1)
        )
        layout.add_widget(self.info_label)
        
        # === ORACIÓN ACTUAL ===
        sentence_container = BoxLayout(orientation='vertical', size_hint=(1, 0.15), spacing=2)
        sentence_container.add_widget(Label(
            text='TEXTO TRADUCIDO:',
            size_hint=(1, 0.3),
            font_size='14sp',
            color=(0.12, 0.23, 0.54, 1),
            bold=True
        ))
        
        self.sentence_label = Label(
            text='La traducción aparecerá aquí...',
            size_hint=(1, 0.7),
            font_size='18sp',
            color=(0.2, 0.2, 0.2, 1),
            text_size=(Window.width - 40, None),
            halign='center',
            valign='middle'
        )
        self.sentence_label.bind(size=self.sentence_label.setter('text_size'))
        sentence_container.add_widget(self.sentence_label)
        layout.add_widget(sentence_container)
        
        # === BOTONES ===
        btn_layout = BoxLayout(size_hint=(1, 0.15), spacing=10)
        
        # Columna izquierda
        left_buttons = BoxLayout(orientation='vertical', size_hint=(0.5, 1), spacing=5)
        
        self.start_btn = Button(
            text='🎥 INICIAR',
            background_color=(0, 0.75, 0.78, 1),  # #00C0C7
            background_normal='',
            font_size='14sp',
            bold=True,
            color=(1, 1, 1, 1)
        )
        self.start_btn.bind(on_press=self.start_capture)
        left_buttons.add_widget(self.start_btn)
        
        self.stop_btn = Button(
            text='⏹️ DETENER',
            background_color=(0.12, 0.23, 0.54, 1),  # #1E3A8A
            background_normal='',
            font_size='14sp',
            bold=True,
            disabled=True,
            color=(1, 1, 1, 1)
        )
        self.stop_btn.bind(on_press=self.stop_capture)
        left_buttons.add_widget(self.stop_btn)
        
        # Columna derecha
        right_buttons = BoxLayout(orientation='vertical', size_hint=(0.5, 1), spacing=5)
        
        self.clear_btn = Button(
            text='🗑️ LIMPIAR',
            background_color=(0.53, 0.81, 0.92, 1),  # #87CEEB
            background_normal='',
            font_size='14sp',
            color=(0.12, 0.23, 0.54, 1)  # Texto azul oscuro
        )
        self.clear_btn.bind(on_press=self.clear_sentence)
        right_buttons.add_widget(self.clear_btn)
        
        self.speak_btn = Button(
            text='🔊 LEER',
            background_color=(1, 0.84, 0, 1),  # #FFD700
            background_normal='',
            font_size='14sp',
            color=(0.12, 0.23, 0.54, 1)  # Texto azul oscuro
        )
        self.speak_btn.bind(on_press=self.speak_sentence)
        right_buttons.add_widget(self.speak_btn)
        
        btn_layout.add_widget(left_buttons)
        btn_layout.add_widget(right_buttons)
        layout.add_widget(btn_layout)
        
        self.add_widget(layout)
    
    # Los métodos restantes (start_capture, stop_capture, etc.) se mantienen igual
    # que en la versión anterior que te proporcioné, solo cambia los colores en:
    
    def update_ui_based_on_result(self, result, status):
        """Actualiza la UI con los nuevos colores"""
        
        if status == 'success':
            pred_type = result.get('type', 'unknown')
            prediction_text = result.get('prediction', '')
            
            if pred_type == 'static':
                self.prediction_label.text = f"🔤 LETRA: {prediction_text}"
                self.prediction_label.color = (0, 0.75, 0.78, 1)  # Turquesa
            else:
                self.prediction_label.text = f"👋 PALABRA: {prediction_text}"
                self.prediction_label.color = (0.12, 0.23, 0.54, 1)  # Azul oscuro
            
            self.progress_bar.value = 0
        
        elif status == 'accumulating':
            counter = result.get('counter', '0/0')
            progress = result.get('progress', 0)
            
            self.prediction_label.text = f"⏳ Detectando... {counter}"
            self.prediction_label.color = (1, 0.84, 0, 1)  # Amarillo
            self.progress_bar.value = progress
        
        elif status == 'detecting':
            self.prediction_label.text = "👋 Moviendo manos para palabras..."
            self.prediction_label.color = (0.53, 0.81, 0.92, 1)  # Azul claro
            self.progress_bar.value = 0
        
        elif status == 'waiting':
            self.prediction_label.text = "✋ Mostrar las manos a la cámara"
            self.prediction_label.color = (0.6, 0.6, 0.6, 1)
            self.progress_bar.value = 0
        
        elif status == 'error':
            error_msg = result.get('message', 'Error')
            self.prediction_label.text = f"❌ {error_msg}"
            self.prediction_label.color = (1, 0.3, 0.3, 1)
            self.progress_bar.value = 0

    # Los demás métodos (start_capture, stop_capture, clear_sentence, speak_sentence, 
    # update, update_camera_display, volver, cleanup_resources, on_leave, on_enter)
    # se mantienen exactamente igual que en la versión anterior que te proporcioné
    
    def start_capture(self, instance):
        """Inicia la captura de cámara"""
        try:
            if self.camera is None:
                for camera_index in [0, 1, 2]:
                    self.camera = cv2.VideoCapture(camera_index)
                    if self.camera.isOpened():
                        print(f"✅ Cámara {camera_index} abierta")
                        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.camera.set(cv2.CAP_PROP_FPS, 30)
                        break
                    else:
                        self.camera = None
                
                if self.camera is None:
                    self.prediction_label.text = "❌ No se pudo abrir la cámara"
                    self.prediction_label.color = (1, 0.3, 0.3, 1)
                    return
            
            self.is_recording = True
            self.start_btn.disabled = True
            self.stop_btn.disabled = False
            
            self.event = Clock.schedule_interval(self.update, self.update_interval)
            
            self.prediction_label.text = "🔄 Iniciando detección..."
            self.prediction_label.color = (0, 0.75, 0.78, 1)
            self.info_label.text = "Coloca las manos frente a la cámara"
            
        except Exception as e:
            self.prediction_label.text = f"❌ Error: {str(e)}"
            self.prediction_label.color = (1, 0.3, 0.3, 1)
    
    def stop_capture(self, instance):
        """Detiene la captura"""
        self.is_recording = False
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
        
        if self.event:
            self.event.cancel()
            self.event = None
        
        self.prediction_label.text = "⏹️ Cámara detenida"
        self.prediction_label.color = (0.6, 0.6, 0.6, 1)
        self.info_label.text = "Presiona INICIAR para reanudar"
        self.progress_bar.value = 0
    
    def clear_sentence(self, instance):
        """Limpia la oración actual"""
        self.translator.reset()
        self.sentence_label.text = 'La traducción aparecerá aquí...'
        self.prediction_label.text = '🧹 Texto reiniciado'
        self.prediction_label.color = (0.53, 0.81, 0.92, 1)
        self.progress_bar.value = 0
    
    def speak_sentence(self, instance):
        """Lee la oración en voz alta"""
        if self.is_speaking:
            return
            
        if not self.translator.sentence:
            self.prediction_label.text = "📝 No hay texto para leer"
            self.prediction_label.color = (1, 0.84, 0, 1)
            return
        
        texto = ' '.join(self.translator.sentence)
        if texto.strip() and self.tts_engine:
            try:
                self.is_speaking = True
                self.speak_btn.disabled = True
                self.prediction_label.text = "🔊 Leyendo texto..."
                self.prediction_label.color = (0, 0.75, 0.78, 1)
                
                def do_tts():
                    try:
                        self.tts_engine.say(texto)
                        self.tts_engine.runAndWait()
                    except Exception as e:
                        print(f"Error en TTS: {e}")
                    finally:
                        Clock.schedule_once(self.enable_speak_button)
                
                import threading
                tts_thread = threading.Thread(target=do_tts)
                tts_thread.daemon = True
                tts_thread.start()
                
            except Exception as e:
                print(f"❌ Error en TTS: {e}")
                self.prediction_label.text = "❌ Error al leer"
                self.prediction_label.color = (1, 0.3, 0.3, 1)
                self.enable_speak_button()
    
    def enable_speak_button(self, dt=None):
        """Reactiva el botón de hablar"""
        self.is_speaking = False
        self.speak_btn.disabled = False
        self.prediction_label.text = "Listo para detectar"
        self.prediction_label.color = (0.12, 0.23, 0.54, 1)
    
    def update(self, dt):
        """Actualiza el frame de la cámara"""
        if not self.is_recording or self.camera is None:
            return
        
        try:
            ret, frame = self.camera.read()
            if not ret:
                self.prediction_label.text = "❌ Error leyendo cámara"
                self.prediction_label.color = (1, 0.3, 0.3, 1)
                return
            
            result = self.translator.process_frame(frame)
            status = result.get('status', 'unknown')
            
            self.update_ui_based_on_result(result, status)
            self.update_camera_display(frame)
            
            current_sentence = ' '.join(self.translator.sentence)
            if current_sentence:
                self.sentence_label.text = current_sentence
                
        except Exception as e:
            print(f"❌ Error en update: {e}")
            self.prediction_label.text = f"❌ Error: {str(e)}"
            self.prediction_label.color = (1, 0.3, 0.3, 1)
    
    def update_camera_display(self, frame):
        """Actualiza la visualización de la cámara"""
        try:
            frame_resized = cv2.resize(frame, (640, 480))
            buf = cv2.flip(frame_resized, 0).tobytes()
            texture = Texture.create(
                size=(frame_resized.shape[1], frame_resized.shape[0]), 
                colorfmt='bgr'
            )
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img_widget.texture = texture
        except Exception as e:
            print(f"❌ Error actualizando display: {e}")
    
    def volver(self, instance):
        """Vuelve a la pantalla principal"""
        self.cleanup_resources()
        self.manager.transition = SlideTransition(direction='right')
        self.manager.current = 'principal'
    
    def cleanup_resources(self):
        """Limpia todos los recursos"""
        if self.is_speaking and self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
            self.is_speaking = False
        
        if self.is_recording:
            self.stop_capture(None)
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        if self.event:
            self.event.cancel()
            self.event = None
    
    def on_leave(self):
        """Se ejecuta cuando se abandona la pantalla"""
        self.cleanup_resources()
    
    def on_enter(self):
        """Se ejecuta cuando se entra a la pantalla"""
        self.prediction_label.text = "Presiona INICIAR para comenzar"
        self.prediction_label.color = (0.12, 0.23, 0.54, 1)
        self.info_label.text = ""
        self.progress_bar.value = 0
        self.is_speaking = False
        self.speak_btn.disabled = False