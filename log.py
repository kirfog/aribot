import itertools
import logging


class ColorInjectedFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "color"):
            record.color = ""
        if not hasattr(record, "reset"):
            record.reset = ""
        return True


handler = logging.StreamHandler()
handler.addFilter(ColorInjectedFilter())
formatter = logging.Formatter("%(color)s[%(levelname)s] %(name)s %(message)s%(reset)s")
handler.setFormatter(formatter)
logger_root = logging.getLogger()
logger_root.addHandler(handler)
logger_root.setLevel(logging.INFO)


class ColoredLogger:
    _bright_colors = [
        48,
        13,
        75,
        39,
        112,
        170,
        208,
        214,
        11,
        45,
        13,
        14,
        46,
        51,
        81,
        82,
        118,
        123,
        154,
        165,
        190,
        196,
        201,
        202,
        220,
        226,
    ]
    _color_pool = itertools.cycle(_bright_colors)

    def __init__(self, name, color=None):

        if color is None:
            color_code = next(self._color_pool)
        else:
            color_code = color

        self.color = f"\033[38;5;{color_code}m"
        self.reset = "\033[0m"

        base_logger = logging.getLogger(name)
        self._adapter = logging.LoggerAdapter(
            base_logger, {"color": self.color, "reset": self.reset}
        )

    def info(self, msg, *args, **kwargs):
        self._adapter.info(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._adapter.error(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._adapter.warning(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._adapter.debug(msg, *args, **kwargs)
