from src.scripts.limpieza_ecv import run as run_limpieza
def main():
    print("Inicio pipeline\n")

    run_limpieza()

    print("\nPipeline finalizado")

if __name__ == "__main__":
    main()
