import sys
import logging

from .agent import CognitiveAgent
from .config import setup_logging
from .ui.cli import run_cli
from .ui.webui import run_web_ui
from .ui.voice import VoiceMode

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    agent = CognitiveAgent()
    if agent.persona and agent.persona.essence:
        logger.info("Adam — %s...", agent.persona.essence[:80])
        logger.info("%s behavioral rules loaded", len(agent.persona.behavior_rules))
    else:
        logger.warning("no persona file found, using generic assistant")
    if "--web" in sys.argv:
        run_web_ui(agent)
    elif "--voice" in sys.argv:
        vm = VoiceMode()
        vm.chat_voice_loop(agent)
    else:
        run_cli(agent)


if __name__ == "__main__":
    main()
