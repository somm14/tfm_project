import os

from src.scripts.bronze_to_silver import run as run_limpieza
from src.scripts.model_pipeline import run as run_modelo


def main():
    print('Inicio pipeline\n')

    run_limpieza()

    print('\nLimpieza finalizada.')

    run_modelo()

    print('\nPipeline finalizado')

if __name__ == '__main__':
    main()