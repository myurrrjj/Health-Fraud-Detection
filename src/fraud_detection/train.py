from .modeling import train_and_save_model


def main() -> None:
    bundle = train_and_save_model()
    print("Provider model trained and saved.")
    for name, value in bundle["metrics"].items():
        print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")


if __name__ == "__main__":
    main()
