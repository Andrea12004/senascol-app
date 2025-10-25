"""
PANTALLA_TEXTO.PY - Texto a Señas
Nuevo diseño inspirado en la imagen proporcionada
"""
from kivy.uix.screenmanager import Screen, SlideTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image as KivyImage
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, RoundedRectangle
import requests
import cv2
import os

class LimitedTextInput(TextInput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_chars = 100
    
    def insert_text(self, substring, from_undo=False):
        if len(self.text) + len(substring) > self.max_chars:
            substring = substring[:self.max_chars - len(self.text)]
        return super().insert_text(substring, from_undo=from_undo)

class TextoScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Configurar fondo blanco
        Window.clearcolor = (1, 1, 1, 1)
        
        self.videos = []
        self.current_video_index = 0
        self.cap = None
        self.play_event = None
        
        # Layout principal
        layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
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
            text='Texto a Señas', 
            font_size='24sp', 
            bold=True, 
            size_hint=(0.85, 1),
            color=(0.12, 0.23, 0.54, 1)  # #1E3A8A
        )
        header.add_widget(header_label)
        layout.add_widget(header)
        
        # === ÁREA DE TEXTO ===
        text_container = BoxLayout(orientation='vertical', size_hint=(1, 0.3), spacing=10)
        
        # Label de instrucción
        text_container.add_widget(Label(
            text='Escribe tu mensaje',
            size_hint=(1, 0.15),
            font_size='16sp',
            color=(0.4, 0.4, 0.4, 1)
        ))
        
        # TextInput con fondo azul claro
        input_container = BoxLayout(orientation='vertical', size_hint=(1, 0.7))
        
        self.text_input = LimitedTextInput(
            multiline=True,
            size_hint=(1, 0.85),
            font_size='18sp',
            hint_text='Hola buenos días, como estás',
            background_color=(0.93, 0.95, 0.96, 1),  # Fondo azul muy claro
            foreground_color=(0.2, 0.2, 0.2, 1),
            padding=[15, 15],
            background_normal='',
            background_active='',
            cursor_color=(0.12, 0.23, 0.54, 1)
        )
        self.text_input.bind(text=self.actualizar_contador)
        input_container.add_widget(self.text_input)
        
        # Contador de caracteres
        counter_layout = BoxLayout(size_hint=(1, 0.15), padding=[10, 0])
        counter_layout.add_widget(Label(
            text='',
            size_hint=(0.8, 1)
        ))
        self.contador_label = Label(
            text='0/100',
            size_hint=(0.2, 1),
            font_size='12sp',
            color=(0.6, 0.6, 0.6, 1),
            halign='right'
        )
        counter_layout.add_widget(self.contador_label)
        input_container.add_widget(counter_layout)
        
        text_container.add_widget(input_container)
        layout.add_widget(text_container)
        
        # === BOTÓN TRADUCIR ===
        self.btn_traducir = Button(
            text='Traducir a Señas',
            size_hint=(1, 0.08),
            background_color=(0, 0.75, 0.78, 1),  # #00C0C7
            background_normal='',
            font_size='18sp',
            bold=True,
            color=(1, 1, 1, 1)
        )
        self.btn_traducir.bind(on_press=self.traducir)
        layout.add_widget(self.btn_traducir)
        
        # === ÁREA DE VIDEO ===
        video_container = BoxLayout(orientation='vertical', size_hint=(1, 0.4), spacing=10)
        video_container.add_widget(Label(
            text='Videos de Señas',
            size_hint=(1, 0.1),
            font_size='16sp',
            color=(0.12, 0.23, 0.54, 1)
        ))
        
        self.video_widget = KivyImage(
            size_hint=(1, 0.9),
            allow_stretch=True,
            keep_ratio=True
        )
        video_container.add_widget(self.video_widget)
        layout.add_widget(video_container)
        
        # === ESTADO Y CACHÉ ===
        footer_layout = BoxLayout(size_hint=(1, 0.1), spacing=10)
        
        # Estado
        self.resultado_label = Label(
            text='Escribe un mensaje y presiona "Traducir"',
            size_hint=(0.7, 1),
            font_size='14sp',
            color=(0.4, 0.4, 0.4, 1)
        )
        footer_layout.add_widget(self.resultado_label)
        
        # Info de caché
        cache_layout = BoxLayout(orientation='vertical', size_hint=(0.3, 1), spacing=2)
        cache_layout.add_widget(Label(
            text='Caché',
            size_hint=(1, 0.4),
            font_size='12sp',
            color=(0.6, 0.6, 0.6, 1)
        ))
        self.cache_label = Label(
            text='0 videos',
            size_hint=(1, 0.6),
            font_size='12sp',
            color=(0.12, 0.23, 0.54, 1)
        )
        cache_layout.add_widget(self.cache_label)
        footer_layout.add_widget(cache_layout)
        
        layout.add_widget(footer_layout)
        
        self.add_widget(layout)
        self.actualizar_cache_info()
    
    def actualizar_contador(self, instance, value):
        """Actualiza el contador de caracteres"""
        longitud = len(value)
        self.contador_label.text = f'{longitud}/100'
        
        # Cambiar color si se acerca al límite
        if longitud > 90:
            self.contador_label.color = (1, 0.3, 0.3, 1)
        elif longitud > 70:
            self.contador_label.color = (1, 0.84, 0, 1)  # #FFD700
        else:
            self.contador_label.color = (0.6, 0.6, 0.6, 1)
    
    def traducir(self, instance):
        texto = self.text_input.text.strip()
        if not texto:
            self.resultado_label.text = '⚠️ Escribe algo primero'
            self.resultado_label.color = (1, 0.84, 0, 1)  # Amarillo
            return
        
        if len(texto) > 100:
            self.resultado_label.text = '⚠️ Máximo 100 caracteres'
            self.resultado_label.color = (1, 0.3, 0.3, 1)
            return
        
        self.resultado_label.text = f'🔄 Traduciendo...'
        self.resultado_label.color = (0, 0.75, 0.78, 1)  # Turquesa
        self.btn_traducir.disabled = True
        self.btn_traducir.text = 'Traduciendo...'
        self.btn_traducir.background_color = (0.7, 0.7, 0.7, 1)
        
        # Ejecutar en un hilo para no bloquear la UI
        import threading
        thread = threading.Thread(target=self._traducir_thread, args=(texto,))
        thread.daemon = True
        thread.start()
    
    def _traducir_thread(self, texto):
        """Hilo para la traducción"""
        try:
            response = requests.post(
                'https://traductor-texto.onrender.com/procesar',
                json={'texto': texto},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                self.videos = data.get('videos', [])
                
                if self.videos:
                    Clock.schedule_once(lambda dt: self._mostrar_exito(len(self.videos)))
                    Clock.schedule_once(lambda dt: self.play_video(0))
                else:
                    Clock.schedule_once(lambda dt: self._mostrar_error('No se encontraron videos'))
            else:
                Clock.schedule_once(lambda dt: self._mostrar_error(f'Error: {response.status_code}'))
        except requests.exceptions.Timeout:
            Clock.schedule_once(lambda dt: self._mostrar_error('Tiempo agotado'))
        except Exception as e:
            Clock.schedule_once(lambda dt: self._mostrar_error(f'Error: {str(e)}'))
        finally:
            Clock.schedule_once(lambda dt: self._habilitar_boton())
    
    def _mostrar_exito(self, cantidad):
        """Muestra éxito en la UI"""
        self.resultado_label.text = f'✅ {cantidad} videos listos'
        self.resultado_label.color = (0.2, 0.8, 0.2, 1)
    
    def _mostrar_error(self, mensaje):
        """Muestra error en la UI"""
        self.resultado_label.text = f'❌ {mensaje}'
        self.resultado_label.color = (1, 0.3, 0.3, 1)
    
    def _habilitar_boton(self):
        """Rehabilita el botón de traducción"""
        self.btn_traducir.disabled = False
        self.btn_traducir.text = 'Traducir a Señas'
        self.btn_traducir.background_color = (0, 0.75, 0.78, 1)
    
    def play_video(self, index):
        if index >= len(self.videos):
            self.resultado_label.text = '✅ Reproducción completada'
            self.stop_video()
            return
        
        self.stop_video()
        
        video_id = self.videos[index]['id']
        video_url = f'https://traductor-texto.onrender.com/videos/{video_id}.mp4'
        
        # Caché
        cache_dir = os.path.join(os.path.expanduser('~'), '.senascol_cache')
        os.makedirs(cache_dir, exist_ok=True)
        video_path = os.path.join(cache_dir, f'{video_id}.mp4')
        
        self.resultado_label.text = f'▶️ {video_id} ({index + 1}/{len(self.videos)})'
        
        try:
            if os.path.exists(video_path):
                self._iniciar_reproduccion(video_path)
            else:
                self._descargar_video(video_url, video_path, index)
        except Exception as e:
            print(f"❌ Error: {e}")
            Clock.schedule_once(lambda dt: self.play_next_video(), 1.0)
    
    def _descargar_video(self, video_url, video_path, index):
        """Descarga el video"""
        def descargar():
            try:
                response = requests.get(video_url, stream=True, timeout=15)
                if response.status_code == 200:
                    with open(video_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    Clock.schedule_once(lambda dt: self._iniciar_reproduccion(video_path))
                else:
                    Clock.schedule_once(lambda dt: self.play_next_video())
            except:
                Clock.schedule_once(lambda dt: self.play_next_video())
        
        import threading
        threading.Thread(target=descargar, daemon=True).start()
    
    def _iniciar_reproduccion(self, video_path):
        """Inicia la reproducción del video"""
        try:
            self.cap = cv2.VideoCapture(video_path)
            if self.cap.isOpened():
                self.play_event = Clock.schedule_interval(self.update_frame, 1.0 / 30.0)
            else:
                Clock.schedule_once(lambda dt: self.play_next_video())
        except:
            Clock.schedule_once(lambda dt: self.play_next_video())
    
    def update_frame(self, dt):
        if not self.cap or not self.cap.isOpened():
            self.play_next_video()
            return
        
        ret, frame = self.cap.read()
        if ret:
            # Redimensionar para mejor visualización
            frame_resized = cv2.resize(frame, (640, 480))
            buf = cv2.flip(frame_resized, 0).tobytes()
            texture = Texture.create(size=(frame_resized.shape[1], frame_resized.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.video_widget.texture = texture
        else:
            self.play_next_video()
    
    def play_next_video(self):
        self.stop_video()
        self.current_video_index += 1
        if self.current_video_index < len(self.videos):
            self.play_video(self.current_video_index)
        else:
            self.resultado_label.text = '✅ ¡Completo!'
            self.video_widget.texture = None
    
    def stop_video(self):
        if self.play_event:
            self.play_event.cancel()
            self.play_event = None
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def actualizar_cache_info(self):
        try:
            cache_dir = os.path.join(os.path.expanduser('~'), '.senascol_cache')
            if not os.path.exists(cache_dir):
                self.cache_label.text = '0 videos'
                return
            
            archivos = [f for f in os.listdir(cache_dir) if f.endswith('.mp4')]
            self.cache_label.text = f'{len(archivos)} videos'
        except:
            self.cache_label.text = 'Error'
    
    def volver(self, instance):
        self.stop_video()
        self.manager.transition = SlideTransition(direction='right')
        self.manager.current = 'principal'