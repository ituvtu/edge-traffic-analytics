from src.app import build_config_from_args, run_app


def main() -> None:
    config = build_config_from_args()
    run_app(config)


if __name__ == "__main__":
    main()
