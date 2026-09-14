"""
Configuracion persistida y cifrada de cada cliente.

## Cuidado con `app_name`: es el DIRECTORIO de configuracion

Hasta el 30-jul-2026, `tienda_view`, `perimetrales-view` y
`windows_managers_view` construian `SettingsModel()` **sin argumentos**, asi que
los tres caian en el valor por defecto `windows_managers_view` y **compartian la
misma carpeta de configuracion**. En disco solo existian dos:
`amazonas_view` y `windows_managers_view`.

Consecuencia real, no teorica: los DVR dados de alta, el establecimiento
seleccionado, el interruptor de WhatsApp y `last_inference` eran **los mismos
para los tres clientes**. Eso es la causa raiz de H-14 — el cliente de tienda
arrancaba con `type_inference = "VigilanteAmazonas"` porque **se lo habia
escrito perimetrales** en la configuracion compartida.

Por eso cada cliente debe pasar SU nombre. Para no perder lo ya guardado,
`legacy_app_name` siembra la configuracion nueva a partir de la antigua la
primera vez.
"""

import json
import os
import shutil
from pathlib import Path
from appdirs import user_config_dir
from cryptography.fernet import Fernet


# ── Posicion inicial de las zonas, en el espacio normalizado 0..1000 ──────
# UNICA fuente de verdad: `default_config` las usa para una instalacion nueva
# y los clientes las reusan como respaldo cuando una config YA EXISTENTE no
# trae la clave. Tenerlas duplicadas en el cliente y aqui era pedir que
# derivaran. Al encender una zona tiene que aparecer DENTRO del encuadre, no
# en una esquina degenerada que el operador tenga que ir a buscar.
ZONA_CAJA_DEFECTO = [[200, 400], [500, 400], [500, 700], [200, 700]]
ZONA_SERVICIO_DEFECTO = [[550, 400], [850, 400], [850, 700], [550, 700]]
# La del personal arranca ARRIBA, lejos de las otras dos: en gris y solapada
# de salida seria facil confundirla, y confundirla falsea el conteo.
ZONA_PERSONAL_DEFECTO = [[200, 100], [500, 100], [500, 330], [200, 330]]


def copia_zona(zona):
    """Copia independiente de una zona por defecto, para que quien la lea
    del ajuste y la edite no mute la constante compartida del módulo."""
    return _copia(zona)


def _copia(zona):
    """Copia PROFUNDA de una zona por defecto.

    Los 16 recuadros no pueden compartir la misma lista: mover un vertice en
    uno los moveria en los dieciseis.
    """
    return [list(punto) for punto in zona]


class SettingsModel:

    def __init__(self, app_name='windows_managers_view',
                 filename='config.json',
                 keyfile='cfghwrpoñm,.}ht4780sSDCWAG.key',
                 legacy_app_name=None):

        config_dir = user_config_dir(app_name)

        # Primera ejecucion con nombre propio: se copia lo que hubiera en la
        # carpeta compartida antigua, para que el usuario no pierda sus DVR ni
        # sus preferencias al separar las configuraciones.
        if legacy_app_name and not os.path.isdir(config_dir):
            viejo = user_config_dir(legacy_app_name)
            if os.path.isdir(viejo):
                try:
                    shutil.copytree(viejo, config_dir)
                except Exception:
                    pass          # si falla, se arranca con la config vacia

        self.key_path = os.path.join(config_dir, keyfile)

        os.makedirs(os.path.dirname(self.key_path), exist_ok=True)
        
        if os.path.exists(self.key_path): 
            with open(self.key_path, 'rb') as f: 
                self.key = f.read() 
        else: 
            self.key = Fernet.generate_key() 
            with open(self.key_path, 'wb') as f: f.write(self.key)
        
        self.f = Fernet(self.key)
        
        project_root = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(project_root, ".."))
        
        
        os.makedirs(config_dir, exist_ok=True)
        
        self.file_path = os.path.join(config_dir, filename)

        self.data = {}
        self.load_config()
        
        
        
    def load_config(self): 

        if os.path.exists(self.file_path): 
            print(self.file_path)
            try:
                with open(self.file_path, 'r', encoding='utf-8') as file: 
                    token = file.read()
                    decryted = self.f.decrypt(token)
                    self.data = json.loads(decryted.decode('utf-8'))
                    print('Archivo de configuración creado 📄')
            except json.JSONDecodeError: # Si el archivo está vacío o corrupto, regenerar 
                self.data = self.default_config() 
                self.save_config()
                print('Archivo de configuración cargado ✅')
        else: 
            self.data = self.default_config()
            self.save_config()
                
                
                
                
    def save_config(self):
        json_str = json.dumps(self.data, indent=4)
        token = self.f.encrypt(json_str.encode('utf-8'))
        with open(self.file_path, 'wb') as file: 
            file.write(token)
            
                
                
    def default_config(self):
        return {
            'principal_box': -1,
            'last_inference': None,
            'selected_establishment':  None,
            'boxs_config': [
                    {
                        'hwnd': None,
                        'index': i,
                        'play': False,
                        'inference_play': False,
                        # ROI de personas (amarillo)
                        'roi': [[100,200],[900,100],[900,900],[100,900]],
                        'roi_boolean': True,
                        # Zona azul «Toma de orden» -> caja del comedor
                        'order_zone': _copia(ZONA_CAJA_DEFECTO),
                        'order_zone_boolean': False,
                        # Zona roja «Entrega de plato» -> linea de servicio
                        'delivery_zone': _copia(ZONA_SERVICIO_DEFECTO),
                        'delivery_zone_boolean': False,
                        # Zona gris «Personal»: quien la ocupa NO es cliente
                        'personal_zone': _copia(ZONA_PERSONAL_DEFECTO),
                        'personal_zone_boolean': False,
                        # ETAPA 2: VLM verificador (Qwen2.5-VL) off por defecto
                        'vlm_enabled_boolean': False,
                        # Mapa de calor (overlay del servidor) off por defecto
                        'heatmap_boolean': False,
                        # Clases COCO a trackear (0 = Persona)
                        'track_classes': [0],
                    } for i in range(16)
            ],
            'amount_renderbox': 2,
            'devices': []
        }
    
    
    
    def set(self, key, value): 
        self.data[key] = value 
        self.save_config() 
        
        
        
    def get(self, key, default=None): 
        return self.data.get(key, default)
    
    
    
    def update_box_config(self, index, key, value):
        """ Actualiza o añade una propiedad dentro de un diccionario de boxs_config. """ 
        for box in self.data['boxs_config']: 
     
            if box['index'] == index:
                box[key] = value 
                self.save_config() 
                
                return # Si no existe el índice, opcionalmente lo creamos 
        new_box = {'index': index, key: value} 
        self.data['boxs_config'].append(new_box) 
        self.save_config()
        
        
        
    def get_box_config(self, index):
        for box in self.data['boxs_config']:
            if box['index'] == index:
               return box


    def add_device(self, name, ip, http_port, rtsp_port, user, password):
        device = {
            'name': name,
            'ip': ip,
            'http_port': http_port,
            'rtsp_port': rtsp_port,
            'user': user,
            'password': password,
            'connected': False
        }
        self.data['devices'].append(device)
        self.save_config()


    def get_devices(self):
        return self.data.get('devices', [])


    def update_device_connection(self, index, connected):
        if 0 <= index < len(self.data['devices']):
            self.data['devices'][index]['connected'] = connected
            self.save_config()

    def remove_device(self, index):
        if 0 <= index < len(self.data['devices']):
            self.data['devices'].pop(index)
            self.save_config()

        