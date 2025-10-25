"""
MAIN.PY - Archivo Principal
Pantalla de inicio con nuevo diseño
"""
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle

class PrincipalScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Configurar fondo blanco
        Window.clearcolor = (1, 1, 1, 1)
        
        layout = BoxLayout(orientation='vertical', padding=40, spacing=30)
        
        # === LOGO Y TÍTULO ===
        header = BoxLayout(orientation='vertical', size_hint=(1, 0.3), spacing=15)
        
        # Logo (puedes reemplazar con tu imagen)
        logo = Label(
            text='🤟',
            font_size='80sp',
            size_hint=(1, 0.6)
        )
        header.add_widget(logo)
        
        title = Label(
            text='SeñasCol',
            font_size='42sp',
            bold=True,
            color=(0.12, 0.23, 0.54, 1)  # #1E3A8A
        )
        subtitle = Label(
            text='Traductor de Lengua de Señas Colombiano',
            font_size='16sp',
            color=(0.4, 0.4, 0.4, 1)
        )
        header.add_widget(title)
        header.add_widget(subtitle)
        layout.add_widget(header)
        
        # === BOTONES PRINCIPALES ===
        buttons_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.5), spacing=20)
        
        # Botón: Texto a Señas
        btn_texto = Button(
            text='📝 Texto a Señas\n\nConvierte texto escrito en representaciones visuales\nde lengua de señas colombiana',
            size_hint=(1, 0.5),
            background_color=(0, 0.75, 0.78, 1),  # #00C0C7
            background_normal='',
            font_size='18sp',
            color=(1, 1, 1, 1),
            bold=True,
            padding=(20, 20)
        )
        btn_texto.bind(on_press=self.ir_texto)
        buttons_layout.add_widget(btn_texto)
        
        # Botón: Señas a Texto
        btn_sena = Button(
            text='🤟 Señas a Texto\n\nReconoce gestos de lengua de señas\ny los traduce a texto escrito',
            size_hint=(1, 0.5),
            background_color=(0.12, 0.23, 0.54, 1),  # #1E3A8A
            background_normal='',
            font_size='18sp',
            color=(1, 1, 1, 1),
            bold=True,
            padding=(20, 20)
        )
        btn_sena.bind(on_press=self.ir_sena)
        buttons_layout.add_widget(btn_sena)
        
        layout.add_widget(buttons_layout)
        
        # === FOOTER ===
        footer = Label(
            text='Comunicación sin barreras',
            font_size='14sp',
            color=(0.6, 0.6, 0.6, 1),
            size_hint=(1, 0.1)
        )
        layout.add_widget(footer)
        
        self.add_widget(layout)
    
    def ir_texto(self, instance):
        """Ir a Texto a Señas"""
        self.manager.transition = SlideTransition(direction='left')
        self.manager.current = 'texto'
    
    def ir_sena(self, instance):
        """Ir a Señas a Texto"""
        self.manager.transition = SlideTransition(direction='left')
        self.manager.current = 'sena'

# ========== APP PRINCIPAL ==========
class SenasColApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(PrincipalScreen(name='principal'))
        
        # Importar aquí para evitar ciclos
        from pantalla_texto import TextoScreen
        from pantalla_senas import SenaScreen
        
        sm.add_widget(TextoScreen(name='texto'))
        sm.add_widget(SenaScreen(name='sena'))
        return sm

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 SEÑASCOL - TRADUCTOR LSC")
    print("=" * 60)
    SenasColApp().run()