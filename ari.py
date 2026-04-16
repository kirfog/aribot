import requests
import websocket


class AsteriskARI(object):
    def __init__(self, appname, server_addr, username, password):
        self._appname = appname
        self._req_base = f"http://{server_addr}:8088/ari/"
        self._ws_base = f"ws://{server_addr}:8088/ari/events?app={appname}"
        self._username = username
        self._password = password

    def _send_post_request(self, req_str):
        r = requests.post(req_str, auth=(self._username, self._password))
        return r

    def _send_delete_request(self, req_str):
        r = requests.delete(req_str, auth=(self._username, self._password))
        return r

    def _send_get_request(self, req_str):
        r = requests.get(req_str, auth=(self._username, self._password))
        return r

    def call_originate(self, call_from, call_to, context):
        req_str = (
            self._req_base
            + f"channels?endpoint=PJSIP/{call_from}&extension={call_to}&context={context}&priority=1&api_key={self._username}:{self._password}"
        )
        self._send_post_request(req_str)

    def answer_call(self, channel_id):
        req_str = self._req_base + f"channels/{channel_id}/answer"
        self._send_post_request(req_str)

    def play_sound(self, channel_id, sound_name):
        req_str = (
            self._req_base
            + f"channels/{channel_id}/play?media=sound:{sound_name}&lang=ru"
        )
        self._send_post_request(req_str)

    def record(self, channel_id, file_name, format, secs):
        req_str = (
            self._req_base
            + f"channels/{channel_id}/record?name={file_name}&format={format}&maxDurationSeconds={secs}"
        )
        self._send_post_request(req_str)

    def continue_in_dialplan(
        self, channel_id, context=None, extension=None, priority="s", label=""
    ):
        if context:
            req_str = (
                self._req_base
                + f"channels/{channel_id}/continue?context={context}&extension={extension}&priority={priority}&label={label}&api_key=asterisk:12345777"
            )
        else:
            req_str = self._req_base + f"channels/{channel_id}/continue"
        self._send_post_request(req_str)

    def music_start(self, channel_id, moh_class):
        req_str = self._req_base + f"channels/{channel_id}/moh?mohClass={moh_class}"
        self._send_post_request(req_str)

    def music_stop(self, channel_id):
        req_str = self._req_base + f"channels/{channel_id}/moh"
        self._send_delete_request(req_str)

    def channel_create(self, endpoint, context, extension, priority):
        req_str = (
            self._req_base
            + f"channels?endpoint=PJSIP/{endpoint}\
            &context={context}&extension={extension}&priority={priority}&app={self._appname}&api_key={self._username}:{self._password}"
        )
        r = self._send_post_request(req_str)
        return r.json()

    def channel_variable(self, channel_id, variable, value):
        req_str = (
            self._req_base
            + f"channels/{channel_id}/variable?variable={variable}&value={value}"
        )
        self._send_post_request(req_str)

    def app_channel_create(self, endpoint):
        req_str = (
            self._req_base
            + f"channels?endpoint=PJSIP/{endpoint}&app={self._appname}&api_key={self._username}:{self._password}"
        )
        r = self._send_post_request(req_str)
        return r.json()

    def external_media_channel_create(self, host, port, format, direction):
        req_str = (
            self._req_base
            + f"channels/externalMedia?app={self._appname}&external_host={host}:{port}&format={format}&direction={direction}&api_key={self._username}:{self._password}"
        )
        r = self._send_post_request(req_str)
        return r.json()

    def snoop_channel_create(self, channel_id, snoop_id, spy, whisper="none"):
        req_str = (
            self._req_base
            + f"channels/{channel_id}/snoop/{snoop_id}?app={self._appname}&spy={spy}&whisper={whisper}&api_key={self._username}:{self._password}"
        )
        r = self._send_post_request(req_str)
        return r.json()

    def channel_delete(self, channel_id):
        req_str = (
            self._req_base
            + f"channels/{channel_id}?api_key={self._username}:{self._password}"
        )
        self._send_delete_request(req_str)

    def channel_get(self, channel_id):
        req_str = (
            self._req_base
            + f"channels/{channel_id}?api_key={self._username}:{self._password}"
        )
        r = self._send_get_request(req_str)
        return r.json()

    def channels_get(self):
        req_str = self._req_base + f"channels?api_key={self._username}:{self._password}"
        r = self._send_get_request(req_str)
        return r.json()

    def channel_hangup(self, channel_id):
        req_str = self._req_base + f"channels/{channel_id}"
        return self._send_delete_request(req_str)

    def bridge_create(self, name):
        req_str = (
            self._req_base
            + f"bridges?type=mixing&name={name}&api_key={self._username}:{self._password}"
        )
        r = self._send_post_request(req_str)
        return r.json()

    def bridge_play_sound(self, bridge_id, sound_name):
        req_str = (
            self._req_base
            + f"bridges/{bridge_id}/play?media=sound:{sound_name}&lang=ru"
        )
        self._send_post_request(req_str)

    def bridge_add_channel(self, bridge_id, channel_id):
        req_str = (
            self._req_base
            + f"bridges/{bridge_id}/addChannel?channel={channel_id}&api_key={self._username}:{self._password}"
        )
        self._send_post_request(req_str)

    def bridge_delete(self, bridge_id):
        req_str = (
            self._req_base
            + f"bridges/{bridge_id}?api_key={self._username}:{self._password}"
        )
        self._send_delete_request(req_str)

    def bridge_get(self, bridge_id):
        req_str = (
            self._req_base
            + f"bridges/{bridge_id}?api_key={self._username}:{self._password}"
        )
        r = self._send_get_request(req_str)
        return r.json()

    def bridge_destroy(self, bridge_id):
        req_str = self._req_base + f"bridges/{bridge_id}"
        return self._send_delete_request(req_str)

    def create_websocket(self):
        url = self._ws_base + f"&api_key={self._username}:{self._password}"
        ws = websocket.create_connection(url)
        return ws
