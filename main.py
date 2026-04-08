from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
import requests, json

API = "http://10.20.36.159:5000"   # ← replace with your computer's IP
token = None

class LoginScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        layout = BoxLayout(orientation='vertical', padding=30, spacing=15)
        layout.add_widget(Label(text='Heart Disease App', font_size=24, size_hint_y=None, height=60))
        self.username = TextInput(hint_text='Username', multiline=False, size_hint_y=None, height=44)
        self.password = TextInput(hint_text='Password', password=True, multiline=False, size_hint_y=None, height=44)
        login_btn  = Button(text='Login',    size_hint_y=None, height=44, on_press=self.login)
        reg_btn    = Button(text='Register', size_hint_y=None, height=44, on_press=self.register)
        self.msg   = Label(text='', size_hint_y=None, height=40)
        for w in [self.username, self.password, login_btn, reg_btn, self.msg]:
            layout.add_widget(w)
        self.add_widget(layout)

    def login(self, *_):
        global token
        r = requests.post(f'{API}/login',
                          json={'username': self.username.text,
                                'password': self.password.text})
        if r.status_code == 200:
            token = r.json()['token']
            self.manager.current = 'home'
        else:
            self.msg.text = r.json().get('error', 'Login failed')

    def register(self, *_):
        r = requests.post(f'{API}/register',
                          json={'username': self.username.text,
                                'password': self.password.text})
        self.msg.text = r.json().get('message', r.json().get('error', ''))

class HomeScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        layout.add_widget(Label(text='Enter Patient Data', font_size=20, size_hint_y=None, height=50))
        fields = ['Age','Sex(0=M,1=F)','ChestPainType(0-3)','RestingBP',
                  'Cholesterol','FastingBS(0/1)','RestingECG(0-2)',
                  'MaxHR','ExerciseAngina(0/1)','Oldpeak','ST_Slope(0-2)']
        self.inputs = []
        for f in fields:
            ti = TextInput(hint_text=f, multiline=False, size_hint_y=None, height=38)
            layout.add_widget(ti)
            self.inputs.append(ti)
        btn = Button(text='Predict', size_hint_y=None, height=50, on_press=self.predict)
        self.result = Label(text='', size_hint_y=None, height=60)
        hist_btn = Button(text='View History', size_hint_y=None, height=44,
                          on_press=lambda *_: setattr(self.manager, 'current', 'history'))
        layout.add_widget(btn)
        layout.add_widget(self.result)
        layout.add_widget(hist_btn)
        sv = ScrollView()
        sv.add_widget(layout)
        self.add_widget(sv)

    def predict(self, *_):
        try:
            features = [float(i.text) for i in self.inputs]
            r = requests.post(f'{API}/predict',
                              json={'features': features},
                              headers={'Authorization': f'Bearer {token}'})
            d = r.json()
            self.result.text = f"{d['result']}  —  Risk: {d['risk_pct']}%"
        except Exception as e:
            self.result.text = f'Error: {e}'

class HistoryScreen(Screen):
    def on_enter(self):
        self.clear_widgets()
        layout = BoxLayout(orientation='vertical', padding=20, spacing=8)
        layout.add_widget(Label(text='Prediction History', font_size=20, size_hint_y=None, height=50))
        r = requests.get(f'{API}/history',
                         headers={'Authorization': f'Bearer {token}'})
        for p in r.json():
            layout.add_widget(Label(
                text=f"{p['date'][:10]}  |  {p['result']}  ({p['risk']}%)",
                size_hint_y=None, height=36))
        back = Button(text='Back', size_hint_y=None, height=44,
                      on_press=lambda *_: setattr(self.manager, 'current', 'home'))
        layout.add_widget(back)
        sv = ScrollView()
        sv.add_widget(layout)
        self.add_widget(sv)

class HeartApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(LoginScreen(name='login'))
        sm.add_widget(HomeScreen(name='home'))
        sm.add_widget(HistoryScreen(name='history'))
        return sm

HeartApp().run()