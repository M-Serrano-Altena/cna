from pprint import pformat
from typing import Any, Optional


class Color:
    """Terminal color codes used to format printed output.

    Attributes:
        PURPLE (str): ANSI escape code for purple text.
        CYAN (str): ANSI escape code for cyan text.
        DARKCYAN (str): ANSI escape code for dark cyan text.
        BLUE (str): ANSI escape code for blue text.
        GREEN (str): ANSI escape code for green text.
        YELLOW (str): ANSI escape code for yellow text.
        RED (str): ANSI escape code for red text.
        BOLD (str): ANSI escape code for bold text.
        UNDERLINE (str): ANSI escape code for underlined text.
        END (str): ANSI escape code to reset formatting.
    """
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class Symbol:
    """Unicode symbols used as prefixes for different message types.

    Attributes:
        DATA (str): Symbol for data-related messages.
        CONFIG (str): Symbol for configuration-related messages.
        WARNING (str): Symbol for warnings.
        EXCEPTION (str): Symbol for exceptions.
        START (str): Symbol used to denote the start of a process.
        LOGS (str): Symbol used for log table output.
        HIGH_SCORE (str): Symbol for high-score / best-model notifications.
        INFO (str): Symbol for informational messages.
    """
    DATA = "📂"
    CONFIG = "📝"
    WARNING = "⚠️"
    EXCEPTION = "🚨"
    START = "💥"
    LOGS = "📊"
    HIGH_SCORE = "🏆"
    INFO = "ℹ️"


def _print(
        obj: Any,
        symbol: str,
        color: Optional[str] = None,
        title: Optional[str] = None,
        symbol_border: bool = False,
        pretty_format: bool = True
) -> None:
    """Format and print a message to standard output with optional styling.

    Args:
        obj (Any): The object or text to print. Will be converted to a
            formatted string; complex objects use pprint.pformat when
            pretty_format is True.
        symbol (str): A short symbol or emoji to prefix each printed line.
        color (Optional[str]): ANSI color code to wrap the message. If None,
            no color code is applied.
        title (Optional[str]): Optional title displayed before the object.
        symbol_border (bool): If True, surround the message with a repeated
            line of the symbol for visual emphasis.
        pretty_format (bool): If True, pretty-print complex objects using
            pprint.pformat with a depth of 3.

    Returns:
        None: This function prints directly to stdout and does not return a
        value.
    """
    if pretty_format:
        obj = pformat(obj, depth=3)
    if title is not None:
        title = f"{title}\n"
    if color is None:
        color = ""
    txt = f"{symbol}\t{color}{Color.BOLD}{title if title is not None else ''}{Color.END}{color}{obj}{Color.END}"
    txt = txt.replace('\n', f'\n{symbol}\t')
    if symbol_border:
        symbol_border_str = symbol * 50
        print(f"{symbol_border_str}\n{txt}\n{symbol_border_str}")
    else:
        print(txt)


def _print_info(obj: Any, symbol: str, title: Optional[str] = None) -> None:
    """Convenience wrapper to print informational messages in yellow.

    Args:
        obj (Any): Object or text to print.
        symbol (str): Symbol/emoji to prefix the message.
        title (Optional[str]): Optional title displayed before the object.

    Returns:
        None
    """
    _print(obj, symbol, Color.YELLOW, title)


def print_start(obj: Any, title: Optional[str] = None) -> None:
    """Print a highlighted start message surrounded by a symbol border.

    This is used to mark the beginning of a process or important event.

    Args:
        obj (Any): Message or object to print.
        title (Optional[str]): Optional title displayed above the message.

    Returns:
        None
    """
    _print(obj, Symbol.START, Color.BLUE, title, symbol_border=True)


def print_logs(logs: Any, title: Optional[str] = None) -> None:
    """Pretty-print a mapping of log keys and values in a tabular style.

    Floats are formatted to four decimal places. The resulting table is
    printed as a single block and not passed through pprint.

    Args:
        logs (Mapping[str, Any]): Dictionary-like object containing log
            entries to display.
        title (Optional[str]): Optional title displayed above the table.

    Returns:
        None
    """
    res = ""
    for k, v in logs.items():
        if isinstance(v, float):
            v = f"{v:.4f}"
        res += f"\t{k:15s}:\t{v}\n"
    _print(res, Symbol.LOGS, Color.BLUE, title, pretty_format=False)


def print_exception(obj: Exception) -> None:
    """Print an exception message with high visibility.

    The message is printed in red and surrounded by a symbol border. The
    provided exception object will be converted to a string for display.

    Args:
        obj (Exception): Exception instance or message to display.

    Returns:
        None
    """
    _print(obj, Symbol.WARNING, Color.RED, "EXCEPTION:\n", symbol_border=True)


def print_warn(obj: Any, title: Optional[str] = None) -> None:
    """Print a warning message in red.

    Args:
        obj (Any): Message or object to print as a warning.
        title (Optional[str]): Optional title displayed before the warning.

    Returns:
        None
    """
    _print(obj, Symbol.WARNING, Color.RED, title)


def print_info_data(obj: Any, title: Optional[str] = None) -> None:
    """Print data-related informational message with a data symbol.

    Args:
        obj (Any): Data or message to print.
        title (Optional[str]): Optional title displayed before the data.

    Returns:
        None
    """
    _print_info(obj, Symbol.DATA, title)


def print_info_config(obj: Any, title: Optional[str] = None) -> None:
    """Print configuration-related information with a config symbol.

    Args:
        obj (Any): Configuration or message to print.
        title (Optional[str]): Optional title displayed before the config.

    Returns:
        None
    """
    _print_info(obj, Symbol.CONFIG, title)


def print_info_best_model(obj: Any, title: Optional[str] = None) -> None:
    """Print a best-model/high-score notification in purple.

    Args:
        obj (Any): Message or object describing the achievement.
        title (Optional[str]): Optional title displayed above the message.

    Returns:
        None
    """
    _print(obj, Symbol.HIGH_SCORE, color=Color.PURPLE, title=title)


def print_info(obj: Any, title: Optional[str] = None) -> None:
    """Print a generic informational message without color.

    Args:
        obj (Any): Message or object to display.
        title (Optional[str]): Optional title displayed before the message.

    Returns:
        None
    """
    _print(obj, Symbol.INFO, color=None, title=title)
