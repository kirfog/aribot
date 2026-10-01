import requests
import websocket


class AsteriskARI(object):
    def __init__(self, appname, server_addr, username, password):
        self._appname = appname
        self._req_base = f"http://{server_addr}:8088/ari/"
        self._ws_base = f"ws://{server_addr}:8088/ari/events?app={appname}"
        self._username = username
        self._password = password
        self.api_key = {"api_key": f"{self._username}:{self._password}"}

    def _send_post_request(self, endpoint, params=None):
        url = f"{self._req_base}{endpoint}"
        full_params = {**self.api_key, **(params or {})}
        return requests.post(
            url, params=full_params, auth=(self._username, self._password)
        )

    def _send_delete_request(self, endpoint, params=None):
        url = f"{self._req_base}{endpoint}"
        full_params = {**self.api_key, **(params or {})}
        return requests.delete(
            url, params=full_params, auth=(self._username, self._password)
        )

    def _send_get_request(self, endpoint, params=None):
        url = f"{self._req_base}{endpoint}"
        full_params = {**self.api_key, **(params or {})}
        return requests.get(
            url, params=full_params, auth=(self._username, self._password)
        )

    def call_originate(self, call_from, call_to, context):
        endpoint = "channels"
        params = {
            "endpoint": f"PJSIP/{call_from}",
            "extension": call_to,
            "context": context,
            "priority": 1,
        }

        self._send_post_request(endpoint, params=params)

    def answer_call(self, channel_id):
        endpoint = f"channels/{channel_id}/answer"
        self._send_post_request(endpoint)

    def play_sound(self, channel_id, sound_name):
        endpoint = f"channels/{channel_id}/play"
        params = {"media": f"sound:{sound_name}", "lang": "ru"}
        self._send_post_request(endpoint, params=params)

    def record(self, channel_id, file_name, format, secs):
        endpoint = f"channels/{channel_id}/record"
        params = {"name": file_name, "format": format, "maxDurationSeconds": secs}
        self._send_post_request(endpoint, params=params)

    def continue_in_dialplan(
        self, channel_id, context=None, extension=None, priority="s", label=""
    ):
        endpoint = f"channels/{channel_id}/continue"
        if context:
            params = {
                "context": context,
                "extension": extension,
                "priority": priority,
                "label": label,
            }
        else:
            params = {}
        self._send_post_request(endpoint, params=params)

    def music_start(self, channel_id, moh_class):
        endpoint = f"channels/{channel_id}/moh"
        params = {
            "mohClass": moh_class,
        }
        self._send_post_request(endpoint, params=params)

    def music_stop(self, channel_id):
        endpoint = f"channels/{channel_id}/moh"
        self._send_delete_request(endpoint)

    def channel_create(self, endpoint, context, extension, priority):
        endpoint = "channels"
        params = {
            "endpoint": f"PJSIP/{endpoint}",
            "context": context,
            "extension": extension,
            "priority": priority,
            "app": self._appname,
        }
        r = self._send_post_request(endpoint, params=params)
        return r.json()

    def channel_variable(self, channel_id, variable, value):
        endpoint = f"channels/{channel_id}/variable"
        params = {"variable": variable, "value": value}
        self._send_post_request(endpoint, params=params)

    def app_channel_create(self, endpoint):
        endpoint = "channels"
        params = {
            "endpoint": f"PJSIP/{endpoint}",
            "app": self._appname,
        }
        r = self._send_post_request(endpoint, params=params)
        return r.json()

    def external_media_channel_create(self, host, port, format, direction):
        endpoint = "channels/externalMedia"
        params = {
            "app": self._appname,
            "external_host": f"{host}:{port}",
            "format": format,
            "direction": direction,
        }
        r = self._send_post_request(endpoint, params=params)
        return r.json()

    def external_ws_media_channel_create(self, host, port, format, direction):
        endpoint = "channels/externalMedia"
        params = {
            "app": self._appname,
            "external_host": f"{host}:{port}",
            "format": format,
            "direction": direction,
            "transport": "websocket",
            "connection_type": "client",
        }
        r = self._send_post_request(endpoint, params=params)
        return r.json()

    def snoop_channel_create(self, channel_id, snoop_id, spy, whisper="none"):
        endpoint = f"channels/{channel_id}/snoop/{snoop_id}"
        params = {
            "app": self._appname,
            "spy": spy,
            "whisper": whisper,
        }
        r = self._send_post_request(endpoint, params=params)
        return r.json()

    def channel_delete(self, channel_id):
        endpoint = f"channels/{channel_id}"
        self._send_delete_request(endpoint)

    def channel_get(self, channel_id):
        endpoint = f"channels/{channel_id}"
        r = self._send_get_request(endpoint)
        return r.json()

    def channels_get(self):
        endpoint = "channels"
        r = self._send_get_request(endpoint)
        return r.json()

    def channel_hangup(self, channel_id):
        endpoint = f"channels/{channel_id}"
        self._send_delete_request(endpoint)

    def bridge_create(self, name):
        endpoint = "bridges"
        params = {
            "type": "mixing",
            "name": name,
        }
        r = self._send_post_request(endpoint, params=params)
        return r.json()

    def bridge_play_sound(self, bridge_id, sound_name):
        endpoint = f"bridges/{bridge_id}/play"
        params = {"media": f"sound:{sound_name}", "lang": "ru"}
        self._send_post_request(endpoint, params=params)

    def bridge_add_channel(self, bridge_id, channel_id):
        endpoint = f"bridges/{bridge_id}/addChannel"
        params = {
            "channel": channel_id,
        }
        self._send_post_request(endpoint, params=params)

    def bridge_delete(self, bridge_id):
        endpoint = f"bridges/{bridge_id}"
        self._send_delete_request(endpoint)

    def bridge_get(self, bridge_id):
        endpoint = f"bridges/{bridge_id}"
        r = self._send_get_request(endpoint)
        return r.json()

    def bridge_destroy(self, bridge_id):
        endpoint = f"bridges/{bridge_id}"
        self._send_delete_request(endpoint)

    def create_websocket(self):
        url = f"{self._ws_base}&api_key={self._username}:{self._password}"
        ws = websocket.create_connection(url)
        return ws
