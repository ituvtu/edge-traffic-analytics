import logging

from dotenv import load_dotenv

load_dotenv()

from src.app import build_config_from_args, run_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    config = build_config_from_args()
    run_app(config)


if __name__ == "__main__":
    main()
